from __future__ import annotations

import json
import subprocess
from pathlib import Path

from examples.hyperliquid import daily_cross_exchange_pipeline as module


def _write_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "daily.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": "daily_cross_exchange_config_v1",
                "profile_id": "skhynix",
                "duration_seconds": 2,
                "continuous_collection": True,
                "collector": {
                    "host": "local",
                    "remote_root": str(tmp_path / "collector"),
                    "python_executable": "python",
                    "segment_duration_seconds": 2,
                },
                "local": {
                    "root": str(tmp_path / "local"),
                    "staging_root": str(tmp_path / "staging"),
                },
                "postprocess": {
                    "profile": "basis-research",
                    "python_executable": "python",
                },
                "simulation": {
                    "remote_root": str(tmp_path / "simulated-remote"),
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return config_path


def _pipeline(tmp_path: Path, run_id: str) -> module.DailyCrossExchangePipeline:
    config = module.load_config(_write_config(tmp_path))
    return module.DailyCrossExchangePipeline(config, run_id, simulate=True)


def test_local_simulation_completes_all_stages(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    assert (
        module.main(
            [
                "run",
                "--config",
                str(config_path),
                "--run-id",
                "20260819-skhynix",
                "--simulate",
            ]
        )
        == 0
    )

    pipeline = _pipeline(tmp_path, "20260819-skhynix")
    status = pipeline.status()
    assert status["state"] == "complete"
    assert all(status["stages"][stage]["passes"] for stage in module.STAGES)
    assert (pipeline.raw_campaign / "campaign_manifest.json").is_file()
    assert (pipeline.working_copy / "timeline_index.csv").is_file()
    assert (pipeline.postprocess_output / "pipeline_manifest.json").is_file()
    assert pipeline.verify()["passes"] is True


def test_successful_run_is_reused_for_same_run_id(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    argv = [
        "run",
        "--config",
        str(config_path),
        "--run-id",
        "20260819-skhynix",
        "--simulate",
    ]
    assert module.main(argv) == 0
    assert module.main(argv) == 0

    pipeline = _pipeline(tmp_path, "20260819-skhynix")
    events = [
        json.loads(line)
        for line in pipeline.event_path.read_text(encoding="utf-8").splitlines()
    ]
    reused = {row["stage"] for row in events if row["event"] == "stage_reused"}
    assert reused == set(module.STAGES)


def test_pull_fails_closed_when_source_hash_changes(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path, "20260819-skhynix")
    pipeline.collect()
    raw = (
        pipeline.remote_campaign
        / "segments"
        / "segment_0001"
        / pipeline.profile_id
        / "sample"
        / "raw.gz"
    )
    raw.write_bytes(b"tampered")

    try:
        pipeline.pull()
    except module.PipelineError as exc:
        assert "inventory mismatch" in str(exc)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("tampered source must fail pull")

    assert pipeline.status()["stages"]["pull"]["passes"] is False
    assert not pipeline.raw_campaign.exists()


def test_remote_collect_reads_remote_evidence_without_local_path(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = module.load_config(_write_config(tmp_path))
    config["collector"]["host"] = "tokyo-collector"
    config["collector"]["remote_root"] = "/var/lib/hftbacktest/daily"
    pipeline = module.DailyCrossExchangePipeline(
        config,
        "20260819-skhynix",
    )
    manifest = {
        "campaign_id": "20260819-skhynix",
        "profiles": ["skhynix"],
        "network_collection_complete": True,
        "passes": True,
    }
    status = {
        "state": "complete",
        "phase": "collection_only_complete",
    }
    inventory = [
        {
            "path": "campaign_manifest.json",
            "bytes": 123,
            "sha256": "a" * 64,
        }
    ]
    outputs = {
        "collection.log": "",
        "remote_campaign_manifest.log": json.dumps(manifest),
        "remote_run_status.log": json.dumps(status),
        "remote_inventory.log": json.dumps(inventory),
    }
    commands = []

    def fake_run(command: list[str], log_name: str):
        commands.append((command, log_name))
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=outputs[log_name],
            stderr="",
        )

    monkeypatch.setattr(pipeline, "_run_command", fake_run)

    result = pipeline.collect()

    assert result["stages"]["collect"]["passes"] is True
    evidence = json.loads(
        (
            pipeline.manifests / "remote_campaign_manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert evidence["campaign_manifest"] == manifest
    assert evidence["run_status"] == status
    assert evidence["source_inventory"] == inventory
    assert [log_name for _, log_name in commands] == [
        "collection.log",
        "remote_campaign_manifest.log",
        "remote_run_status.log",
        "remote_inventory.log",
    ]
    assert "systemd-run" in commands[0][0][-1]


def test_command_builders_preserve_stage_boundaries(tmp_path: Path) -> None:
    supervisor = module.build_supervisor_postprocess_command(
        python_executable="python",
        campaign_dir=tmp_path / "campaign",
        campaign_id="run-1",
        profile_id="skhynix",
        duration_seconds=10,
        segment_duration_seconds=10,
        task_id="task-1",
    )
    assert "--postprocess-only" in supervisor
    assert "--clean-output" not in supervisor

    postprocess = module.build_postprocess_command(
        python_executable="python",
        campaign_dir=tmp_path / "campaign",
        output_dir=tmp_path / "output",
        profile_id="skhynix",
        postprocess_profile="basis-research",
        task_id="task-2",
    )
    assert postprocess[1:4] == ["-m", module.POSTPROCESS_MODULE, "run"]
    assert "--profile" in postprocess
    assert "basis-research" in postprocess
