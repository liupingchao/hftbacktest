#!/usr/bin/env python3
"""Daily cross-exchange collection, transfer, and postprocess orchestrator.

The production path delegates collection and timeline construction to the
existing supervisor, then delegates dataset construction to the
cross_exchange-postprocess module.  ``--simulate`` is a deterministic local
fixture path for testing orchestration without network access.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import gzip
import json
import os
import shlex
import shutil
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SUPERVISOR_SCRIPT = (
    PROJECT_ROOT
    / "examples"
    / "hyperliquid"
    / "cross_exchange_collection_supervisor.py"
)
POSTPROCESS_MODULE = "examples.hyperliquid.cross_exchange_postprocess"
CONFIG_SCHEMA_VERSION = "daily_cross_exchange_config_v1"
PIPELINE_SCHEMA_VERSION = "daily_cross_exchange_pipeline_v1"
STAGES = ("collect", "pull", "timeline", "postprocess", "report")

try:
    from .cross_exchange_postprocess.contracts import (
        atomic_write_json,
        atomic_write_text,
        inventory_files,
        inventory_fingerprint,
        sha256_file,
    )
    from .cross_exchange_postprocess.pipeline import (
        inspect_campaign,
        validate_pipeline_output,
    )
    from .cross_exchange_postprocess.reporting import write_reports
    from .cross_exchange_symbol_registry import get_symbol_profile
except ImportError:  # pragma: no cover - direct script execution
    sys.path.insert(0, str(PROJECT_ROOT))
    from examples.hyperliquid.cross_exchange_postprocess.contracts import (
        atomic_write_json,
        atomic_write_text,
        inventory_files,
        inventory_fingerprint,
        sha256_file,
    )
    from examples.hyperliquid.cross_exchange_postprocess.pipeline import (
        inspect_campaign,
        validate_pipeline_output,
    )
    from examples.hyperliquid.cross_exchange_postprocess.reporting import write_reports
    from examples.hyperliquid.cross_exchange_symbol_registry import get_symbol_profile


class PipelineError(RuntimeError):
    """Raised when a pipeline stage must fail closed."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_run_id(profile_id: str) -> str:
    return f"{datetime.now(timezone.utc):%Y%m%d}-{profile_id}"


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise PipelineError(f"expected JSON object: {path}")
    return payload


def _resolve_path(value: str, *, config_path: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def load_config(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    payload = _read_json(path)
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise PipelineError(
            f"unsupported config schema: {payload.get('schema_version')!r}"
        )
    profile_id = str(payload.get("profile_id", "")).strip().lower()
    if not profile_id:
        raise PipelineError("config.profile_id is required")
    profile = get_symbol_profile(profile_id)
    duration = float(payload.get("duration_seconds", 0))
    if duration <= 0:
        raise PipelineError("config.duration_seconds must be positive")
    collector = payload.get("collector", {})
    local = payload.get("local", {})
    postprocess = payload.get("postprocess", {})
    if not isinstance(collector, dict) or not isinstance(local, dict):
        raise PipelineError("collector and local must be objects")
    if not isinstance(postprocess, dict):
        raise PipelineError("postprocess must be an object")
    postprocess_profile = str(postprocess.get("profile", "dataset"))
    if postprocess_profile not in {"dataset", "basis-research"}:
        raise PipelineError(
            "postprocess.profile must be dataset or basis-research"
        )
    payload["_config_path"] = str(path)
    payload["_profile"] = {
        "profile_id": profile.profile_id,
        "binance_symbol": profile.binance_symbol,
        "hyperliquid_coin": profile.hyperliquid_coin,
        "target_symbol": profile.target_symbol,
    }
    payload["_local_root"] = str(
        _resolve_path(
            str(
                local.get(
                    "root",
                    "local_live_analysis/daily_cross_exchange",
                )
            ),
            config_path=path,
        )
    )
    staging_root = local.get("staging_root", "/tmp/hftbacktest-daily-cross-exchange")
    payload["_staging_root"] = str(_resolve_path(str(staging_root), config_path=path))
    return payload


def _safe_run_id(run_id: str) -> str:
    allowed = (
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789._-"
    )
    if not run_id or any(char not in allowed for char in run_id):
        raise PipelineError(f"invalid run id: {run_id!r}")
    return run_id


def _jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


@contextmanager
def _file_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="ascii") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise PipelineError(f"pipeline lock is active: {path}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _copy_tree(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise PipelineError(f"missing campaign directory: {source}")
    if destination.exists():
        raise PipelineError(f"staging directory already exists: {destination}")
    shutil.copytree(source, destination)


def _publish_directory(staging: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise PipelineError(f"destination already exists: {destination}")
    os.replace(staging, destination)


def build_supervisor_postprocess_command(
    *,
    python_executable: str,
    campaign_dir: Path,
    campaign_id: str,
    profile_id: str,
    duration_seconds: float,
    segment_duration_seconds: float,
    task_id: str,
    supervisor_script: Path | str = SUPERVISOR_SCRIPT,
    collection_only: bool = False,
    continuous_collection: bool = False,
) -> list[str]:
    command = [
        python_executable,
        str(supervisor_script),
        "--output-dir",
        str(campaign_dir),
        "--campaign-id",
        campaign_id,
        "--profiles",
        profile_id,
        "--total-duration-seconds",
        str(duration_seconds),
        "--segment-duration-seconds",
        str(segment_duration_seconds),
        "--task-id",
        task_id,
    ]
    command.append("--collection-only" if collection_only else "--postprocess-only")
    if continuous_collection:
        command.append("--continuous-collection")
    return command


def build_postprocess_command(
    *,
    python_executable: str,
    campaign_dir: Path,
    output_dir: Path,
    profile_id: str,
    postprocess_profile: str,
    task_id: str,
    resume: bool = False,
) -> list[str]:
    return [
        python_executable,
        "-m",
        POSTPROCESS_MODULE,
        "resume" if resume else "run",
        "--campaign-dir",
        str(campaign_dir),
        "--output-dir",
        str(output_dir),
        "--symbol-profile",
        profile_id,
        "--profile",
        postprocess_profile,
        "--task-id",
        task_id,
    ]


class DailyCrossExchangePipeline:
    """Stateful, resumable implementation of one configured run."""

    def __init__(
        self,
        config: dict[str, Any],
        run_id: str,
        *,
        simulate: bool = False,
    ) -> None:
        self.config = config
        self.profile_id = str(config["profile_id"])
        self.run_id = _safe_run_id(run_id)
        self.simulate = bool(simulate or config.get("simulation", {}).get("enabled", False))
        self.root = Path(config["_local_root"]) / self.profile_id / self.run_id
        self.logs = self.root / "logs"
        self.manifests = self.root / "manifests"
        self.status_path = self.root / "run_status.json"
        self.event_path = self.logs / "pipeline.jsonl"
        self.lock_path = (
            Path(config["_local_root"]) / self.profile_id / ".daily-pipeline.lock"
        )
        simulation = config.get("simulation", {})
        simulation_root = simulation.get(
            "remote_root",
            str(Path(config["_staging_root"]) / "simulated_remote"),
        )
        self.simulation_remote_root = Path(str(simulation_root)).expanduser()
        self.remote_root = str(config.get("collector", {}).get("remote_root", ""))

    @property
    def raw_campaign(self) -> Path:
        return self.root / "raw_campaign"

    @property
    def working_copy(self) -> Path:
        return self.root / "campaign_working_copy"

    @property
    def postprocess_output(self) -> Path:
        return self.root / "preprocess" / "postprocess_output"

    @property
    def remote_campaign(self) -> Path:
        return (
            self.simulation_remote_root
            / self.profile_id
            / self.run_id
            / "campaign"
        )

    @property
    def local_collector_campaign(self) -> Path:
        collector_root = Path(str(self.config["collector"]["remote_root"])).expanduser()
        return collector_root / self.profile_id / self.run_id / "campaign"

    def _status(self) -> dict[str, Any]:
        if not self.status_path.is_file():
            return {
                "schema_version": PIPELINE_SCHEMA_VERSION,
                "run_id": self.run_id,
                "profile_id": self.profile_id,
                "simulation": self.simulate,
                "state": "created",
                "stages": {},
            }
        return _read_json(self.status_path)

    def _write_status(self, **updates: Any) -> dict[str, Any]:
        status = self._status()
        status.update(updates)
        status["updated_at"] = utc_now()
        atomic_write_json(self.status_path, status)
        return status

    def _event(self, event: str, **fields: Any) -> None:
        _jsonl(
            self.event_path,
            {
                "timestamp": utc_now(),
                "event": event,
                "run_id": self.run_id,
                "profile_id": self.profile_id,
                "simulation": self.simulate,
                **fields,
            },
        )

    def _stage_state(self, stage: str) -> dict[str, Any]:
        return self._status().get("stages", {}).get(stage, {})

    def _begin_stage(self, stage: str) -> None:
        status = self._status()
        stages = dict(status.get("stages", {}))
        stages[stage] = {
            "status": "running",
            "passes": False,
            "started_at": utc_now(),
        }
        self._write_status(
            state=f"{stage}_running",
            current_stage=stage,
            stages=stages,
        )
        self._event("stage_started", stage=stage)

    def _finish_stage(
        self,
        stage: str,
        *,
        passes: bool,
        artifacts: list[dict[str, Any]] | None = None,
        **fields: Any,
    ) -> None:
        status = self._status()
        stages = dict(status.get("stages", {}))
        previous = dict(stages.get(stage, {}))
        previous.update(
            {
                "status": "complete" if passes else "failed",
                "passes": passes,
                "finished_at": utc_now(),
                **fields,
            }
        )
        if artifacts is not None:
            previous["artifacts"] = artifacts
        stages[stage] = previous
        self._write_status(
            state=f"{stage}_complete" if passes else f"{stage}_failed",
            current_stage=None if passes else stage,
            stages=stages,
        )
        self._event(
            "stage_finished",
            stage=stage,
            passes=passes,
            error=fields.get("error", ""),
        )

    def _reuse_stage(self, stage: str, **fields: Any) -> bool:
        state = self._stage_state(stage)
        if state.get("status") != "complete" or state.get("passes") is not True:
            return False
        self._event("stage_reused", stage=stage, **fields)
        return True

    def preflight(self) -> dict[str, Any]:
        with _file_lock(self.lock_path):
            self.root.mkdir(parents=True, exist_ok=True)
            self.logs.mkdir(parents=True, exist_ok=True)
            self.manifests.mkdir(parents=True, exist_ok=True)
            self._write_status(
                state="preflight_running",
                config_path=self.config["_config_path"],
                symbol=self.config["_profile"],
                simulation=self.simulate,
            )
            self._event("preflight_started")
            collector = self.config.get("collector", {})
            if not collector.get("host"):
                raise PipelineError("collector.host is required")
            self._write_status(
                state="preflight_complete",
                preflight={
                    "passes": True,
                    "collector_host": collector["host"],
                    "remote_root": collector.get("remote_root", ""),
                    "local_root": str(self.root),
                },
            )
            self._event("preflight_finished", passes=True)
            return self._status()

    def _ensure_preflight(self) -> None:
        if self.status_path.is_file():
            return
        self.root.mkdir(parents=True, exist_ok=True)
        self.logs.mkdir(parents=True, exist_ok=True)
        self.manifests.mkdir(parents=True, exist_ok=True)
        self._write_status(
            state="preflight_complete",
            config_path=self.config["_config_path"],
            symbol=self.config["_profile"],
            simulation=self.simulate,
            preflight={"passes": True, "implicit": True},
        )

    def _run_command(
        self,
        command: list[str],
        log_name: str,
    ) -> subprocess.CompletedProcess[str]:
        log_path = self.logs / log_name
        self._event("command_started", command=command, log=str(log_path))
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        atomic_write_text(
            log_path,
            "\n".join(
                [
                    f"$ {subprocess.list2cmdline(command)}",
                    "",
                    result.stdout,
                    result.stderr,
                ]
            ),
        )
        self._event(
            "command_finished",
            command=command,
            returncode=result.returncode,
            log=str(log_path),
        )
        if result.returncode != 0:
            raise PipelineError(
                f"command failed ({result.returncode}): {subprocess.list2cmdline(command)}"
            )
        return result

    def _ssh_prefix(self) -> list[str]:
        collector = self.config["collector"]
        return [
            "ssh",
            "-o",
            f"ConnectTimeout={int(collector.get('ssh_connect_timeout_seconds', 15))}",
            str(collector["host"]),
        ]

    def _remote_json(self, path: str, log_name: str) -> dict[str, Any]:
        result = self._run_command(
            [
                *self._ssh_prefix(),
                shlex.join(["cat", path]),
            ],
            log_name,
        )
        payload = json.loads(result.stdout)
        if not isinstance(payload, dict):
            raise PipelineError(f"remote JSON is not an object: {path}")
        return payload

    def _remote_inventory(self, campaign_path: str) -> list[dict[str, Any]]:
        collector = self.config["collector"]
        script = """import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
rows = []
for path in sorted(item for item in root.rglob("*") if item.is_file()):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    rows.append(
        {
            "path": str(path.relative_to(root)),
            "bytes": path.stat().st_size,
            "sha256": digest.hexdigest(),
        }
    )
print(json.dumps(rows, separators=(",", ":"), sort_keys=True))
"""
        result = self._run_command(
            [
                *self._ssh_prefix(),
                shlex.join(
                    [
                        str(collector.get("python_executable", sys.executable)),
                        "-c",
                        script,
                        campaign_path,
                    ]
                ),
            ],
            "remote_inventory.log",
        )
        payload = json.loads(result.stdout)
        if not isinstance(payload, list) or not payload:
            raise PipelineError("remote campaign inventory is empty or invalid")
        records = []
        for record in payload:
            if not isinstance(record, dict):
                raise PipelineError("remote campaign inventory contains invalid record")
            records.append(
                {
                    "path": str(record["path"]),
                    "bytes": int(record["bytes"]),
                    "sha256": str(record["sha256"]),
                }
            )
        return records

    def _simulate_collect(self) -> Path:
        campaign = self.remote_campaign
        if campaign.is_dir():
            try:
                inspect_campaign(campaign, self.profile_id)
                return campaign
            except Exception:
                shutil.rmtree(campaign)
        campaign.parent.mkdir(parents=True, exist_ok=True)
        staging = campaign.parent / f".{campaign.name}.staging"
        if staging.exists():
            shutil.rmtree(staging)
        segment = staging / "segments" / "segment_0001"
        profile = segment / self.profile_id
        sample = profile / "sample"
        sample.mkdir(parents=True, exist_ok=True)
        raw_path = sample / "raw.gz"
        with gzip.open(raw_path, "wt", encoding="utf-8", newline="") as handle:
            handle.write('{"local_ts_ns":1000000000,"source":"simulated"}\n')
        timeline_path = profile / "common_l2_timeline.csv.gz"
        with gzip.open(timeline_path, "wt", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["local_ts_ns", "binance_mid", "hyperliquid_mid"])
            writer.writerow(["1000000000", "100.0", "100.1"])
        atomic_write_json(
            profile / "common_l2_timeline_manifest.json",
            {
                "schema_version": "common_l2_timeline_v1",
                "passes": True,
                "timeline_row_count": 1,
                "simulation": True,
            },
        )
        atomic_write_json(
            segment / "segment_manifest.json",
            {
                "schema_version": "cross_exchange_segment_v1",
                "segment_id": "segment_0001",
                "passes": True,
                "simulation": True,
            },
        )
        (staging / "timeline_index.csv").write_text(
            "segment_id,profile_id,row_count\nsegment_0001,"
            f"{self.profile_id},1\n",
            encoding="utf-8",
        )
        campaign_manifest = {
            "schema_version": "cross_exchange_collection_campaign_v1",
            "task_id": f"daily-{self.run_id}",
            "campaign_id": self.run_id,
            "execution_mode": "collection_only",
            "profiles": [self.profile_id],
            "requested_total_duration_seconds": float(
                self.config["duration_seconds"]
            ),
            "segment_duration_seconds": float(
                self.config["duration_seconds"]
            ),
            "segments": [
                {
                    "segment_id": "segment_0001",
                    "requested_duration_seconds": float(
                        self.config["duration_seconds"]
                    ),
                }
            ],
            "cross_segment_continuity_claimed": False,
            "degraded_intervals": [],
            "network_collection_complete": True,
            "postprocess_pending": True,
            "simulation": True,
            "passes": True,
        }
        atomic_write_json(staging / "campaign_manifest.json", campaign_manifest)
        atomic_write_json(
            staging / "run_status.json",
            {
                "state": "complete",
                "phase": "collection_only_complete",
                "passes": True,
                "simulation": True,
            },
        )
        records = inventory_files(
            staging,
            excluded_relative_paths=("campaign_manifest.json", "run_status.json"),
        )
        campaign_manifest["source_inventory"] = records
        campaign_manifest["source_inventory_sha256"] = inventory_fingerprint(records)
        atomic_write_json(staging / "campaign_manifest.json", campaign_manifest)
        os.replace(staging, campaign)
        return campaign

    def collect(self) -> dict[str, Any]:
        self._ensure_preflight()
        with _file_lock(self.lock_path):
            if self._reuse_stage("collect"):
                return self._status()
            self._begin_stage("collect")
            try:
                if self.simulate:
                    campaign = self._simulate_collect()
                else:
                    campaign = self._collect_remote()
                collector_host = str(self.config["collector"]["host"])
                if self.simulate or collector_host in {"local", "localhost"}:
                    manifest = self._validate_collection(campaign)
                    status = _read_json(campaign / "run_status.json")
                    source_inventory = inventory_files(campaign)
                else:
                    campaign_path = str(campaign)
                    manifest = self._remote_json(
                        f"{campaign_path}/campaign_manifest.json",
                        "remote_campaign_manifest.log",
                    )
                    status = self._remote_json(
                        f"{campaign_path}/run_status.json",
                        "remote_run_status.log",
                    )
                    self._validate_collection_documents(manifest, status)
                    source_inventory = self._remote_inventory(campaign_path)
                evidence = {
                    "schema_version": "daily_cross_exchange_remote_collection_v1",
                    "run_id": self.run_id,
                    "profile_id": self.profile_id,
                    "campaign_dir": str(campaign),
                    "campaign_manifest": manifest,
                    "run_status": status,
                    "source_inventory": source_inventory,
                    "source_inventory_sha256": inventory_fingerprint(
                        source_inventory
                    ),
                    "passes": True,
                    "simulation": self.simulate,
                }
                atomic_write_json(
                    self.manifests / "remote_campaign_manifest.json",
                    evidence,
                )
                self._finish_stage(
                    "collect",
                    passes=True,
                    campaign_dir=str(campaign),
                    campaign_id=manifest.get("campaign_id"),
                    source_inventory_sha256=evidence[
                        "source_inventory_sha256"
                    ],
                )
                return self._status()
            except Exception as exc:
                self._finish_stage("collect", passes=False, error=str(exc))
                raise

    def _campaign_manifest(self, campaign: Path) -> dict[str, Any]:
        path = campaign / "campaign_manifest.json"
        if not path.is_file():
            raise PipelineError(f"missing campaign manifest: {path}")
        return _read_json(path)

    def _validate_collection_documents(
        self,
        manifest: dict[str, Any],
        status: dict[str, Any],
    ) -> None:
        if manifest.get("passes") is not True:
            raise PipelineError("remote campaign manifest did not pass")
        if manifest.get("network_collection_complete") is not True:
            raise PipelineError("network collection is not complete")
        if self.profile_id not in manifest.get("profiles", []):
            raise PipelineError("remote campaign profile does not match config")
        if status.get("state") != "complete" or status.get("passes") is False:
            raise PipelineError(
                f"remote campaign status is not complete: {status.get('state')!r}"
            )

    def _validate_collection(self, campaign: Path) -> dict[str, Any]:
        manifest = self._campaign_manifest(campaign)
        status_path = campaign / "run_status.json"
        if not status_path.is_file():
            raise PipelineError(f"missing campaign status: {status_path}")
        status = _read_json(status_path)
        self._validate_collection_documents(manifest, status)
        return manifest

    def _validate_postprocess_contract(
        self,
        output_dir: Path,
    ) -> dict[str, Any]:
        manifest_path = output_dir / "pipeline_manifest.json"
        if not manifest_path.is_file():
            raise PipelineError(f"missing postprocess manifest: {manifest_path}")
        manifest = _read_json(manifest_path)
        if manifest.get("status") != "complete" or manifest.get("passes") is not True:
            raise PipelineError("postprocess pipeline manifest did not pass")
        if manifest.get("source_immutable") is not True:
            raise PipelineError("postprocess source is not immutable")
        required_stages = ["raw_audit", "r0", "r1"]
        if manifest.get("profile") == "basis-research":
            required_stages.append("basis_dislocation")
        stages = manifest.get("stages", {})
        for stage_id in required_stages:
            state = stages.get(stage_id, {})
            if state.get("status") != "complete" or state.get("passes") is not True:
                raise PipelineError(f"postprocess stage did not pass: {stage_id}")
        accepted_horizons = stages.get("r1", {}).get("summary", {}).get(
            "accepted_primary_horizons_ms",
            [],
        )
        if not accepted_horizons:
            raise PipelineError("postprocess has no accepted primary horizon")
        return manifest

    def _collect_remote(self) -> Path:
        collector = self.config["collector"]
        host = str(collector["host"])
        campaign_path = (
            str(Path(str(collector["remote_root"])) / self.profile_id / self.run_id / "campaign")
        )
        command = build_supervisor_postprocess_command(
            python_executable=str(collector.get("python_executable", sys.executable)),
            campaign_dir=Path(campaign_path),
            campaign_id=self.run_id,
            profile_id=self.profile_id,
            duration_seconds=float(self.config["duration_seconds"]),
            segment_duration_seconds=float(
                collector.get(
                    "segment_duration_seconds",
                    self.config["duration_seconds"],
                )
            ),
            task_id=f"daily-{self.run_id}",
            supervisor_script=str(
                collector.get("supervisor_script", SUPERVISOR_SCRIPT)
            ),
            collection_only=True,
            continuous_collection=bool(
                self.config.get("continuous_collection", False)
            ),
        )
        if host in {"local", "localhost"}:
            self._run_command(command, "collection.log")
            return Path(campaign_path)
        remote_command = [
            "systemd-run",
            "--unit",
            f"hftbacktest-daily-{self.run_id}",
            "--collect",
            "--wait",
            "--",
            *command,
        ]
        ssh_command = [*self._ssh_prefix(), shlex.join(remote_command)]
        self._run_command(ssh_command, "collection.log")
        return Path(campaign_path)

    def pull(self) -> dict[str, Any]:
        self._ensure_preflight()
        with _file_lock(self.lock_path):
            if self._reuse_stage("pull") and self.raw_campaign.is_dir():
                return self._status()
            if not self._stage_state("collect").get("passes"):
                raise PipelineError("pull requires a passing collect stage")
            self._begin_stage("pull")
            staging = (
                Path(self.config["_staging_root"])
                / self.profile_id
                / self.run_id
                / "raw_campaign"
            )
            try:
                if staging.exists():
                    shutil.rmtree(staging)
                evidence_path = (
                    self.manifests / "remote_campaign_manifest.json"
                )
                if not evidence_path.is_file():
                    raise PipelineError(
                        f"missing remote collection evidence: {evidence_path}"
                    )
                evidence = _read_json(evidence_path)
                expected = evidence.get("source_inventory", [])
                if not isinstance(expected, list) or not expected:
                    raise PipelineError(
                        "remote collection evidence has no source inventory"
                    )
                if self.simulate or str(self.config["collector"]["host"]) in {
                    "local",
                    "localhost",
                }:
                    source = (
                        self.remote_campaign
                        if self.simulate
                        else self.local_collector_campaign
                    )
                    _copy_tree(source, staging)
                else:
                    staging.mkdir(parents=True, exist_ok=True)
                    collector = self.config["collector"]
                    source = str(
                        Path(str(collector["remote_root"]))
                        / self.profile_id
                        / self.run_id
                        / "campaign"
                    )
                    self._run_command(
                        [
                            "rsync",
                            "-a",
                            "--partial",
                            f"{collector['host']}:{source}/",
                            f"{staging}/",
                        ],
                        "pull.log",
                    )
                manifest = self._campaign_manifest(staging)
                actual = inventory_files(
                    staging,
                )
                if actual != expected:
                    raise PipelineError("pulled campaign inventory mismatch")
                self._validate_collection(staging)
                if self.raw_campaign.exists():
                    raise PipelineError(
                        f"raw campaign already exists and is not reusable: {self.raw_campaign}"
                    )
                _publish_directory(staging, self.raw_campaign)
                transfer = {
                    "schema_version": "daily_cross_exchange_transfer_v1",
                    "run_id": self.run_id,
                    "profile_id": self.profile_id,
                    "source_campaign_id": manifest.get("campaign_id"),
                    "source_inventory": expected,
                    "source_inventory_sha256": inventory_fingerprint(expected),
                    "local_inventory": actual,
                    "local_inventory_sha256": inventory_fingerprint(actual),
                    "passes": True,
                    "simulation": self.simulate,
                }
                atomic_write_json(
                    self.manifests / "local_transfer_manifest.json",
                    transfer,
                )
                self._finish_stage(
                    "pull",
                    passes=True,
                    local_campaign=str(self.raw_campaign),
                    local_inventory_sha256=transfer["local_inventory_sha256"],
                )
                return self._status()
            except Exception as exc:
                if staging.exists():
                    failed = staging.with_name(staging.name + ".failed")
                    if failed.exists():
                        shutil.rmtree(failed)
                    os.replace(staging, failed)
                self._finish_stage("pull", passes=False, error=str(exc))
                raise

    def timeline(self) -> dict[str, Any]:
        self._ensure_preflight()
        with _file_lock(self.lock_path):
            if self._reuse_stage("timeline") and self.working_copy.is_dir():
                return self._status()
            if not self._stage_state("pull").get("passes"):
                raise PipelineError("timeline requires a passing pull stage")
            self._begin_stage("timeline")
            staging = (
                Path(self.config["_staging_root"])
                / self.profile_id
                / self.run_id
                / "campaign_working_copy"
            )
            try:
                if staging.exists():
                    shutil.rmtree(staging)
                _copy_tree(self.raw_campaign, staging)
                if not self.simulate:
                    collector = self.config["collector"]
                    command = build_supervisor_postprocess_command(
                        python_executable=str(
                            self.config.get("postprocess", {}).get(
                                "python_executable",
                                collector.get("python_executable", sys.executable),
                            )
                        ),
                        campaign_dir=staging,
                        campaign_id=self.run_id,
                        profile_id=self.profile_id,
                        duration_seconds=float(self.config["duration_seconds"]),
                        segment_duration_seconds=float(
                            collector.get(
                                "segment_duration_seconds",
                                self.config["duration_seconds"],
                            )
                        ),
                        task_id=f"daily-{self.run_id}-timeline",
                        supervisor_script=SUPERVISOR_SCRIPT,
                    )
                    self._run_command(command, "preprocess.log")
                raw_manifest = self._campaign_manifest(self.raw_campaign)
                inspected = inspect_campaign(staging, self.profile_id)
                if inspected.get("campaign_id") != raw_manifest.get("campaign_id"):
                    raise PipelineError("working copy campaign identity changed")
                _publish_directory(staging, self.working_copy)
                atomic_write_json(self.manifests / "timeline_manifest.json", inspected)
                self._finish_stage(
                    "timeline",
                    passes=True,
                    campaign_dir=str(self.working_copy),
                    source_inventory_sha256=inspected["source_inventory_sha256"],
                    timeline_row_count=inspected["segment_count"],
                )
                return self._status()
            except Exception as exc:
                if staging.exists():
                    failed = staging.with_name(staging.name + ".failed")
                    if failed.exists():
                        shutil.rmtree(failed)
                    os.replace(staging, failed)
                self._finish_stage("timeline", passes=False, error=str(exc))
                raise

    def _simulate_postprocess(self) -> dict[str, Any]:
        output = self.postprocess_output
        if output.exists():
            shutil.rmtree(output)
        output.mkdir(parents=True, exist_ok=True)
        source_records = inventory_files(self.working_copy)
        profile = str(self.config["postprocess"].get("profile", "dataset"))
        stages = {
            "raw_audit": {
                "status": "complete",
                "passes": True,
                "artifacts": [],
                "summary": {
                    "source_file_count": len(source_records),
                    "source_inventory_sha256": inventory_fingerprint(source_records),
                },
            },
            "r0": {
                "status": "complete",
                "passes": True,
                "artifacts": [],
                "summary": {"aggregate_counts": {"timeline_rows": 1}},
            },
            "r1": {
                "status": "complete",
                "passes": True,
                "artifacts": [],
                "summary": {
                    "accepted_primary_horizons_ms": [1000],
                    "horizon_mask_exclusion_count": 0,
                    "cross_epoch_label_count": 0,
                },
            },
        }
        if profile == "basis-research":
            stages["basis_dislocation"] = {
                "status": "complete",
                "passes": True,
                "artifacts": [],
                "summary": {
                    "aggregate_counts": {
                        "state_rows": 1,
                        "book_eligible_rows": 1,
                    },
                    "join_rule": "strict_asof_source_ts_lte_decision_ts",
                    "rolling_closed": "left",
                },
            }
        manifest = {
            "schema_version": "cross_exchange_postprocess_pipeline_v1",
            "task_id": f"daily-{self.run_id}-postprocess",
            "campaign_id": self.run_id,
            "symbol_profile": self.profile_id,
            "profile": profile,
            "campaign_dir": str(self.working_copy),
            "source_immutable": True,
            "status": "complete",
            "passes": True,
            "simulation": True,
            "stages": stages,
            "capability_matrix": {
                "raw_audit": "available",
                "r0_event_store": "simulated",
                "r1_alignment": "simulated",
                "basis_dislocation": (
                    "simulated"
                    if profile == "basis-research"
                    else "not_requested"
                ),
            },
            "claim_boundaries": {
                "public_l2_only": True,
                "exact_fill_or_pnl": "not_established",
            },
            "failures": [],
        }
        atomic_write_json(output / "pipeline_manifest.json", manifest)
        atomic_write_json(
            output / "provenance_lock.json",
            {
                "schema_version": "cross_exchange_provenance_lock_v1",
                "campaign_dir": str(self.working_copy),
                "campaign_id": self.run_id,
                "symbol_profile": self.profile_id,
                "source_inventory_sha256": inventory_fingerprint(source_records),
                "source_file_count": len(source_records),
                "raw_file_count": len(
                    [record for record in source_records if record["path"].endswith("/raw.gz")]
                ),
                "source_inventory": source_records,
                "source_immutable": True,
            },
        )
        write_reports(output, manifest)
        return manifest

    def postprocess(self) -> dict[str, Any]:
        self._ensure_preflight()
        with _file_lock(self.lock_path):
            if self._reuse_stage("postprocess") and self.postprocess_output.is_dir():
                return self._status()
            if not self._stage_state("timeline").get("passes"):
                raise PipelineError("postprocess requires a passing timeline stage")
            self._begin_stage("postprocess")
            try:
                if self.simulate:
                    manifest = self._simulate_postprocess()
                else:
                    output_exists = self.postprocess_output.is_dir()
                    command = build_postprocess_command(
                        python_executable=str(
                            self.config.get("postprocess", {}).get(
                                "python_executable", sys.executable
                            )
                        ),
                        campaign_dir=self.working_copy,
                        output_dir=self.postprocess_output,
                        profile_id=self.profile_id,
                        postprocess_profile=str(
                            self.config["postprocess"].get("profile", "dataset")
                        ),
                        task_id=f"daily-{self.run_id}-postprocess",
                        resume=output_exists,
                    )
                    self._run_command(command, "preprocess.log")
                    manifest = _read_json(
                        self.postprocess_output / "pipeline_manifest.json"
                    )
                self._validate_postprocess_contract(self.postprocess_output)
                validation = validate_pipeline_output(self.postprocess_output)
                if validation["passes"] is not True:
                    raise PipelineError(
                        f"postprocess validation failed: {validation['failures']}"
                    )
                atomic_write_json(
                    self.manifests / "postprocess_manifest.json",
                    {"pipeline": manifest, "validation": validation},
                )
                self._finish_stage(
                    "postprocess",
                    passes=True,
                    output_dir=str(self.postprocess_output),
                    pipeline_manifest_sha256=sha256_file(
                        self.postprocess_output / "pipeline_manifest.json"
                    ),
                    validation=validation,
                )
                return self._status()
            except Exception as exc:
                self._finish_stage("postprocess", passes=False, error=str(exc))
                raise

    def report(self) -> dict[str, Any]:
        self._ensure_preflight()
        with _file_lock(self.lock_path):
            if self._reuse_stage("report") and (self.root / "run_report.md").is_file():
                return self._status()
            self._begin_stage("report")
            try:
                status = self._status()
                lines = [
                    "# Daily Cross-Exchange Pipeline Report",
                    "",
                    f"- Run ID: `{self.run_id}`",
                    f"- Profile: `{self.profile_id}`",
                    f"- Simulation: `{self.simulate}`",
                    f"- Updated: `{utc_now()}`",
                    "",
                    "## Stages",
                    "",
                ]
                for stage in STAGES:
                    state = status.get("stages", {}).get(stage, {})
                    lines.append(
                        f"- `{stage}`: `{state.get('status', 'not_run')}` "
                        f"(passes=`{state.get('passes', False)}`)"
                    )
                lines.extend(
                    [
                        "",
                        "## Outputs",
                        "",
                        f"- Raw campaign: `{self.raw_campaign}`",
                        f"- Working copy: `{self.working_copy}`",
                        f"- Postprocess output: `{self.postprocess_output}`",
                        "",
                        "This report is an orchestration record. The postprocess "
                        "Skill report remains the dataset-level quality report.",
                        "",
                    ]
                )
                atomic_write_text(self.root / "run_report.md", "\n".join(lines))
                self._finish_stage("report", passes=True, report=str(self.root / "run_report.md"))
                self._write_status(state="complete", current_stage=None)
                return self._status()
            except Exception as exc:
                self._finish_stage("report", passes=False, error=str(exc))
                raise

    def run(self, stages: list[str]) -> dict[str, Any]:
        self.preflight()
        for stage in stages:
            if stage not in STAGES:
                raise PipelineError(f"unsupported stage: {stage}")
            if stage == "collect":
                self.collect()
            elif stage == "pull":
                self.pull()
            elif stage == "timeline":
                self.timeline()
            elif stage == "postprocess":
                self.postprocess()
            elif stage == "report":
                self.report()
        return self._status()

    def inspect(self) -> dict[str, Any]:
        campaign = self.working_copy
        if not campaign.is_dir():
            raise PipelineError(f"missing working copy: {campaign}")
        result = inspect_campaign(campaign, self.profile_id)
        atomic_write_json(self.manifests / "timeline_manifest.json", result)
        return result

    def validate_postprocess(self) -> dict[str, Any]:
        self._validate_postprocess_contract(self.postprocess_output)
        return validate_pipeline_output(self.postprocess_output)

    def verify(self) -> dict[str, Any]:
        status = self._status()
        failures = []
        for stage in ("collect", "pull", "timeline", "postprocess"):
            if status.get("stages", {}).get(stage, {}).get("passes") is not True:
                failures.append(f"stage:{stage}")
        if not self.raw_campaign.is_dir():
            failures.append("missing:raw_campaign")
        if not self.working_copy.is_dir():
            failures.append("missing:campaign_working_copy")
        if not self.postprocess_output.is_dir():
            failures.append("missing:postprocess_output")
        if self.postprocess_output.is_dir():
            validation = self.validate_postprocess()
            if validation["passes"] is not True:
                failures.extend(validation["failures"])
        result = {
            "schema_version": "daily_cross_exchange_verify_v1",
            "run_id": self.run_id,
            "profile_id": self.profile_id,
            "failures": failures,
            "passes": not failures,
        }
        atomic_write_json(self.manifests / "verification.json", result)
        self._event("verification_finished", **result)
        return result

    def status(self) -> dict[str, Any]:
        return self._status()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Run one or more pipeline stages.")
    run.add_argument("--config", required=True)
    run.add_argument(
        "--run-id",
        help="Stable run identifier; defaults to YYYYMMDD-profile in UTC.",
    )
    run.add_argument("--stages", default=",".join(STAGES))
    run.add_argument("--simulate", action="store_true")

    for name in (
        "preflight",
        "collect",
        "pull",
        "timeline",
        "postprocess",
        "postprocess-inspect",
        "postprocess-validate",
        "verify",
        "status",
        "report",
    ):
        command = subparsers.add_parser(name)
        command.add_argument("--config", required=True)
        command.add_argument("--run-id")
        command.add_argument("--simulate", action="store_true")

    for name, stage in (
        ("retry-pull", "pull"),
        ("retry-timeline", "timeline"),
        ("retry-postprocess", "postprocess"),
    ):
        command = subparsers.add_parser(name, help=f"Retry the {stage} stage.")
        command.add_argument("--config", required=True)
        command.add_argument("--run-id", required=True)
        command.add_argument("--simulate", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = load_config(Path(args.config))
        run_id = args.run_id or default_run_id(str(config["profile_id"]))
        pipeline = DailyCrossExchangePipeline(
            config,
            run_id,
            simulate=bool(args.simulate),
        )
        if args.command == "run":
            stages = [stage.strip() for stage in args.stages.split(",") if stage.strip()]
            result = pipeline.run(stages)
        elif args.command == "preflight":
            result = pipeline.preflight()
        elif args.command == "collect":
            result = pipeline.collect()
        elif args.command == "pull":
            result = pipeline.pull()
        elif args.command == "timeline":
            result = pipeline.timeline()
        elif args.command == "postprocess":
            result = pipeline.postprocess()
        elif args.command == "postprocess-inspect":
            result = pipeline.inspect()
        elif args.command == "postprocess-validate":
            result = pipeline.validate_postprocess()
        elif args.command == "verify":
            result = pipeline.verify()
        elif args.command == "status":
            result = pipeline.status()
        elif args.command == "report":
            result = pipeline.report()
        else:
            stage = {
                "retry-pull": "pull",
                "retry-timeline": "timeline",
                "retry-postprocess": "postprocess",
            }[args.command]
            result = getattr(pipeline, stage)()
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result.get("passes", True) is True else 5
    except (PipelineError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(json.dumps({"passes": False, "error": str(exc)}, indent=2))
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
