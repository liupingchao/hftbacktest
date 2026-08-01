import importlib.util
import json
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


MODULE_PATH = Path(__file__).with_name("ec2_hunt.py")
SPEC = importlib.util.spec_from_file_location("ec2_hunt", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

ORCHESTRATOR_PATH = Path(__file__).with_name("ec2_hunt_orchestrator.py")
ORCHESTRATOR_SPEC = importlib.util.spec_from_file_location(
    "ec2_hunt_orchestrator", ORCHESTRATOR_PATH
)
ORCHESTRATOR = importlib.util.module_from_spec(ORCHESTRATOR_SPEC)
ORCHESTRATOR_SPEC.loader.exec_module(ORCHESTRATOR)

SPREAD_PATH = Path(__file__).with_name("ec2_spread_hunt.py")
sys.path.insert(0, str(SPREAD_PATH.parent))
SPREAD_SPEC = importlib.util.spec_from_file_location(
    "ec2_spread_hunt", SPREAD_PATH
)
SPREAD = importlib.util.module_from_spec(SPREAD_SPEC)
SPREAD_SPEC.loader.exec_module(SPREAD)

BUILD_PATH = Path(__file__).with_name("ec2_hunt_build.py")
BUILD_SPEC = importlib.util.spec_from_file_location(
    "ec2_hunt_build", BUILD_PATH
)
BUILD = importlib.util.module_from_spec(BUILD_SPEC)
BUILD_SPEC.loader.exec_module(BUILD)


def test_rank_rows_is_stable_and_lowest_first():
    rows = [
        {"instance_type": "b", "instance_id": "2", "metric": 20},
        {"instance_type": "a", "instance_id": "1", "metric": 10},
        {"instance_type": "c", "instance_id": "3", "metric": 20},
    ]
    ranked = MODULE.rank_rows(rows, "metric")
    assert [row["instance_type"] for row in ranked] == ["a", "b", "c"]
    assert [row["rank"] for row in ranked] == [1, 2, 3]


def test_balanced_selection_uses_frozen_weights_and_tie_breaks():
    rows = [
        {
            "instance_type": "c6in.xlarge",
            "instance_id": "i-a",
            "candidate_label": "candidate-01",
            "clock_max_bound_us": 200.0,
            "binance_feed_p50_ns": 20,
            "binance_feed_p99_ns": 10,
            "binance_tick_p99_ns": 30,
            "hyperliquid_tick_p99_ns": 30,
            "benchmark_p99_ns": 30,
        },
        {
            "instance_type": "c6in.xlarge",
            "instance_id": "i-b",
            "candidate_label": "candidate-02",
            "clock_max_bound_us": 100.0,
            "binance_feed_p50_ns": 10,
            "binance_feed_p99_ns": 20,
            "binance_tick_p99_ns": 10,
            "hyperliquid_tick_p99_ns": 10,
            "benchmark_p99_ns": 10,
        },
    ]
    rankings = {
        metric: MODULE.rank_rows(rows, metric)
        for metric in MODULE.BALANCED_WEIGHTS
    }
    selection = MODULE.balanced_selection(rows, rankings)
    assert selection["weights"]["binance_feed_p99_ns"] == 4
    assert selection["winner_instance_id"] == "i-b"
    assert selection["ranking"][0]["weighted_score"] == 14
    assert selection["ranking"][1]["weighted_score"] == 16


def test_spread_group_is_rack_level_and_launches_labeled_candidates(
    monkeypatch, tmp_path
):
    state_path = tmp_path / "state.json"
    state = ORCHESTRATOR.load_state(
        state_path, "0729T001", "run-test", "ap-northeast-1"
    )
    args = SimpleNamespace(
        region="ap-northeast-1",
        run_id="run-test",
        task="0729T001",
        image_id="ami-test",
        subnet_id="subnet-test",
        security_group_id="sg-test",
        iam_instance_profile="profile-test",
        key_name="key-test",
        availability_zone="ap-northeast-1c",
        user_data_file=tmp_path / "user-data.sh",
    )
    calls = []
    ids = iter(["i-01", "i-02"])

    def fake_aws(_region, service, operation, *aws_args, **_kwargs):
        calls.append((service, operation, aws_args))
        if (service, operation) == ("ec2", "run-instances"):
            return SimpleNamespace(stdout=f"{next(ids)}\n")
        return SimpleNamespace(stdout="", returncode=0, stderr="")

    monkeypatch.setattr(SPREAD.control, "aws", fake_aws)
    group = SPREAD.create_spread_group(args, state_path, state)
    candidates = SPREAD.launch_batch(
        args, state_path, state, group, range(1, 3)
    )

    create_call = calls[0]
    assert create_call[:2] == ("ec2", "create-placement-group")
    assert "spread" in create_call[2]
    assert "rack" in create_call[2]
    assert len(candidates) == 2
    assert candidates[0][3] == "candidate-01"
    assert candidates[1][3] == "candidate-02"
    assert state["placement_groups"] == [group]
    assert state["instances"] == ["i-01", "i-02"]


def test_spread_hunt_constants_freeze_seven_c6in_hosts():
    assert SPREAD.CANDIDATE_COUNT == 7
    assert SPREAD.INSTANCE_TYPE == "c6in.xlarge"
    assert SPREAD.DURATION_SECONDS == 900
    assert SPREAD.WARMUP_MESSAGES == 100


def test_failure_cleanup_only_disables_task_candidate_protection(monkeypatch):
    state = {
        "region": "ap-northeast-1",
        "instances": ["i-candidate-1", "i-candidate-2"],
    }
    disabled = []
    cleaned = []

    monkeypatch.setattr(
        SPREAD,
        "disable_termination_protection",
        lambda region, instance_ids: disabled.append(
            (region, list(instance_ids))
        ),
    )
    monkeypatch.setattr(
        SPREAD.control,
        "cleanup",
        lambda state_path, current: cleaned.append((state_path, current)),
    )
    SPREAD.failure_cleanup(Path("state.json"), state)
    assert disabled == [
        (
            "ap-northeast-1",
            ["i-candidate-1", "i-candidate-2"],
        )
    ]
    assert cleaned == [(Path("state.json"), state)]


def test_finalize_winner_protects_before_loser_cleanup_and_excludes_baseline(
    monkeypatch, tmp_path
):
    state_path = tmp_path / "state.json"
    state = ORCHESTRATOR.load_state(
        state_path, "0729T002", "run-test", "ap-northeast-1"
    )
    args = SimpleNamespace(
        region="ap-northeast-1",
        task="0729T002",
        run_id="run-test",
        bucket="bucket-test",
        protected_instance_id="i-awsserver1",
    )
    winner = "i-candidate-07"
    candidates = [
        (
            "c6in.xlarge",
            f"i-candidate-{index:02d}",
            "pg-test",
            f"candidate-{index:02d}",
        )
        for index in range(1, 8)
    ]
    report = {
        "candidate_count": 7,
        "eligible_count": 7,
        "balanced_selection": {"winner_instance_id": winner},
    }
    sequence = []

    def fake_aws(_region, service, operation, *aws_args, **_kwargs):
        sequence.append((service, operation, aws_args))
        if (service, operation) == (
            "ec2",
            "describe-instance-attribute",
        ):
            return SimpleNamespace(
                stdout=json.dumps(
                    {
                        "InstanceId": winner,
                        "DisableApiTermination": {"Value": True},
                    }
                )
            )
        if (service, operation) == ("ec2", "describe-instances"):
            return SimpleNamespace(
                stdout=json.dumps(
                    {
                        "InstanceId": winner,
                        "InstanceType": "c6in.xlarge",
                        "State": "running",
                        "AvailabilityZone": "ap-northeast-1c",
                        "GroupName": "pg-test",
                        "Tags": [],
                    }
                )
            )
        return SimpleNamespace(stdout="", returncode=0, stderr="")

    monkeypatch.setattr(SPREAD.control, "aws", fake_aws)
    monkeypatch.setattr(
        SPREAD,
        "empty_and_delete_bucket",
        lambda _args: sequence.append(("s3", "delete-bucket", ())),
    )
    monkeypatch.setattr(
        SPREAD,
        "protected_instance_receipt",
        lambda _args: {
            "InstanceId": "i-awsserver1",
            "State": "running",
        },
    )
    monkeypatch.setattr(
        SPREAD.control,
        "discover_instance_ids",
        lambda *_args: [winner],
    )
    monkeypatch.setattr(
        SPREAD.control,
        "discover_placement_groups",
        lambda *_args: ["pg-test"],
    )
    monkeypatch.setattr(
        SPREAD.control, "discover_buckets", lambda *_args: []
    )

    receipt = SPREAD.finalize_winner(
        args,
        state_path,
        state,
        candidates,
        "pg-test",
        report,
        {"InstanceId": "i-awsserver1", "State": "running"},
    )

    operations = [
        (service, operation) for service, operation, _args in sequence
    ]
    protect_index = operations.index(
        ("ec2", "modify-instance-attribute")
    )
    first_verify_index = operations.index(
        ("ec2", "describe-instance-attribute")
    )
    terminate_index = operations.index(("ec2", "terminate-instances"))
    bucket_delete_index = operations.index(("s3", "delete-bucket"))
    assert protect_index < first_verify_index < terminate_index
    assert terminate_index < bucket_delete_index

    terminate_args = sequence[terminate_index][2]
    terminated_ids = set(
        terminate_args[terminate_args.index("--instance-ids") + 1 :]
    )
    assert terminated_ids == {
        f"i-candidate-{index:02d}" for index in range(1, 7)
    }
    assert winner not in terminated_ids
    assert "i-awsserver1" not in terminated_ids
    assert receipt["verified"]
    assert receipt["active_task_instances"] == [winner]
    assert len(receipt["loser_instance_ids"]) == 6
    assert receipt["winner_termination_protection_before_cleanup"][
        "DisableApiTermination"
    ]["Value"]


def test_finalize_winner_stops_before_loser_mutation_if_protection_fails(
    monkeypatch, tmp_path
):
    state_path = tmp_path / "state.json"
    state = ORCHESTRATOR.load_state(
        state_path, "0729T002", "run-test", "ap-northeast-1"
    )
    args = SimpleNamespace(
        region="ap-northeast-1",
        task="0729T002",
        run_id="run-test",
    )
    candidates = [
        (
            "c6in.xlarge",
            f"i-candidate-{index:02d}",
            "pg-test",
            f"candidate-{index:02d}",
        )
        for index in range(1, 8)
    ]
    report = {
        "candidate_count": 7,
        "eligible_count": 7,
        "balanced_selection": {
            "winner_instance_id": "i-candidate-07"
        },
    }
    operations = []

    def fake_aws(_region, service, operation, *_args, **_kwargs):
        operations.append((service, operation))
        if (service, operation) == (
            "ec2",
            "describe-instance-attribute",
        ):
            return SimpleNamespace(
                stdout=json.dumps(
                    {"DisableApiTermination": {"Value": False}}
                )
            )
        return SimpleNamespace(stdout="", returncode=0, stderr="")

    monkeypatch.setattr(SPREAD.control, "aws", fake_aws)
    with pytest.raises(
        RuntimeError, match="winner protection did not become active"
    ):
        SPREAD.finalize_winner(
            args,
            state_path,
            state,
            candidates,
            "pg-test",
            report,
            {"InstanceId": "i-awsserver1", "State": "running"},
        )
    assert ("ec2", "terminate-instances") not in operations


def test_validate_summary_rejects_hidden_drop():
    summary = {
        "trace_count": 10,
        "integrity": {
            "complete_required": 10,
            "incomplete_required": 0,
            "duplicate_trace_ids": 0,
            "duplicate_stage_marks": 0,
            "out_of_order": 0,
            "dropped_traces": 1,
        },
    }
    assert not MODULE.validate_summary(summary)


def test_validate_clock_pair_requires_distinct_bounded_samples():
    before = {
        "captured_at_utc": "2026-07-28T00:00:00Z",
        "leap_status": "Normal",
        "error_bound_us": 200.0,
        "error_bound_limit_us": 750.0,
        "gate": "pass",
    }
    after = {
        **before,
        "captured_at_utc": "2026-07-28T00:15:00Z",
        "error_bound_us": 300.0,
    }
    assert MODULE.validate_clock_pair(before, after)
    after["error_bound_us"] = 751.0
    assert not MODULE.validate_clock_pair(before, after)
    after["error_bound_us"] = 300.0
    after["captured_at_utc"] = before["captured_at_utc"]
    assert not MODULE.validate_clock_pair(before, after)


def test_cleanup_plan_unions_state_and_tag_discovery():
    state = {
        "instances": ["i-state"],
        "placement_groups": ["pg-state"],
        "buckets": ["bucket-state"],
    }
    plan = ORCHESTRATOR.cleanup_plan(
        state,
        discovered_instances=["i-tag", "i-state"],
        discovered_groups=["pg-tag", "pg-state"],
    )
    assert plan == {
        "instances": ["i-state", "i-tag"],
        "placement_groups": ["pg-state", "pg-tag"],
        "buckets": ["bucket-state"],
    }


def test_launch_failure_persists_group_and_failure_event(monkeypatch, tmp_path):
    state_path = tmp_path / "state.json"
    state = ORCHESTRATOR.load_state(
        state_path, "0728T072", "run-test", "ap-northeast-1"
    )
    args = SimpleNamespace(
        region="ap-northeast-1",
        run_id="run-test",
        task="0728T072",
        image_id="ami-test",
        subnet_id="subnet-test",
        security_group_id="sg-test",
        iam_instance_profile="profile-test",
        key_name="key-test",
        availability_zone="ap-northeast-1c",
        user_data_file=tmp_path / "user-data.sh",
    )

    def fake_aws(_region, service, operation, *args, **kwargs):
        if (service, operation) == ("ec2", "create-placement-group"):
            return SimpleNamespace(stdout="")
        raise RuntimeError("simulated RunInstances failure")

    monkeypatch.setattr(ORCHESTRATOR, "aws", fake_aws)
    with pytest.raises(RuntimeError, match="simulated RunInstances failure"):
        ORCHESTRATOR.launch_candidates(
            args, state_path, state, ["c7i.xlarge"]
        )

    persisted = ORCHESTRATOR.load_state(
        state_path, "0728T072", "run-test", "ap-northeast-1"
    )
    assert persisted["placement_groups"] == [
        "hft-run-test-c7i-xlarge"
    ]
    assert persisted["instances"] == []
    assert persisted["events"][-1]["kind"] == "launch_failure"
    assert persisted["events"][-1]["value"]["instance_type"] == "c7i.xlarge"


def test_parse_upload_receipt_rejects_failed_upload():
    stdout = (
        'noise\n{"label":"smoke","instance_id":"i-test",'
        '"archive_sha256":"abc","archive_bytes":123,"upload_status":1}\n'
    )
    with pytest.raises(RuntimeError, match="remote upload failed"):
        ORCHESTRATOR.parse_upload_receipt(stdout)


def test_create_bucket_records_plan_before_api_call(monkeypatch, tmp_path):
    state_path = tmp_path / "state.json"
    state = ORCHESTRATOR.load_state(
        state_path, "0728T073", "run-test", "ap-northeast-1"
    )
    args = SimpleNamespace(
        region="ap-northeast-1",
        task="0728T073",
        run_id="run-test",
        bucket="bucket-test",
    )
    sequence = []

    def fake_record(_state_path, _state, kind, value):
        sequence.append(("record", kind, value))
        if kind == "buckets":
            _state["buckets"].append(value)

    def fake_aws(_region, service, operation, *args, **kwargs):
        sequence.append(("aws", service, operation))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(ORCHESTRATOR, "record", fake_record)
    monkeypatch.setattr(ORCHESTRATOR, "aws", fake_aws)
    ORCHESTRATOR.create_bucket(args, state_path, state)

    assert sequence[0] == ("record", "buckets", "bucket-test")
    assert sequence[1] == ("record", "bucket_planned", "bucket-test")
    assert sequence[2] == ("aws", "s3api", "create-bucket")


def test_cleanup_delete_failure_is_not_reported_as_success(
    monkeypatch, tmp_path
):
    state_path = tmp_path / "state.json"
    state = ORCHESTRATOR.load_state(
        state_path, "0728T073", "run-test", "ap-northeast-1"
    )
    state["placement_groups"] = ["pg-test"]

    monkeypatch.setattr(
        ORCHESTRATOR, "discover_instance_ids", lambda *_args: []
    )
    monkeypatch.setattr(
        ORCHESTRATOR, "discover_placement_groups", lambda *_args: []
    )
    monkeypatch.setattr(ORCHESTRATOR, "discover_buckets", lambda *_args: [])

    def fake_aws(_region, service, operation, *args, **kwargs):
        if (service, operation) == ("ec2", "delete-placement-group"):
            return SimpleNamespace(
                returncode=1, stdout="", stderr="simulated delete failure"
            )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(ORCHESTRATOR, "aws", fake_aws)
    with pytest.raises(RuntimeError, match="cleanup verification failed"):
        ORCHESTRATOR.cleanup(state_path, state)

    persisted = json.loads(state_path.read_text())
    assert persisted["events"][-1]["kind"] == "cleanup_failure"
    assert not persisted["events"][-1]["value"]["verified"]


def test_cleanup_head_bucket_access_denied_fails_closed(
    monkeypatch, tmp_path
):
    state_path = tmp_path / "state.json"
    state = ORCHESTRATOR.load_state(
        state_path, "0728T074", "run-test", "ap-northeast-1"
    )
    state["buckets"] = ["bucket-test"]
    monkeypatch.setattr(
        ORCHESTRATOR, "discover_instance_ids", lambda *_args: []
    )
    monkeypatch.setattr(
        ORCHESTRATOR, "discover_placement_groups", lambda *_args: []
    )
    monkeypatch.setattr(ORCHESTRATOR, "discover_buckets", lambda *_args: [])

    def fake_aws(_region, service, operation, *args, **kwargs):
        if (service, operation) == ("s3api", "head-bucket"):
            return SimpleNamespace(
                returncode=255, stdout="", stderr="403 AccessDenied"
            )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(ORCHESTRATOR, "aws", fake_aws)
    with pytest.raises(RuntimeError, match="cleanup verification failed"):
        ORCHESTRATOR.cleanup(state_path, state)
    persisted = json.loads(state_path.read_text())
    result = persisted["events"][-1]["value"]
    assert not result["verified"]
    assert any("AccessDenied" in error for error in result["errors"])


def test_execute_runs_cleanup_after_early_failure(monkeypatch, tmp_path):
    args = SimpleNamespace(
        state_file=tmp_path / "state.json",
        task="0728T073",
        run_id="run-test",
        region="ap-northeast-1",
    )
    cleanup_calls = []

    def fail_create_bucket(*_args):
        raise RuntimeError("simulated create failure")

    def fake_cleanup(*_args):
        cleanup_calls.append(True)

    monkeypatch.setattr(ORCHESTRATOR, "create_bucket", fail_create_bucket)
    monkeypatch.setattr(ORCHESTRATOR, "cleanup", fake_cleanup)
    with pytest.raises(RuntimeError, match="simulated create failure"):
        ORCHESTRATOR.execute(args)
    assert cleanup_calls == [True]


def test_execute_signal_interrupt_records_and_cleans_up(monkeypatch, tmp_path):
    args = SimpleNamespace(
        state_file=tmp_path / "state.json",
        task="0728T074",
        run_id="run-test",
        region="ap-northeast-1",
    )
    handlers = {}
    cleanup_calls = []

    def fake_signal(signum, handler):
        previous = handlers.get(signum, signal.SIG_DFL)
        handlers[signum] = handler
        return previous

    def interrupt_create_bucket(*_args):
        handlers[signal.SIGTERM](signal.SIGTERM, None)

    monkeypatch.setattr(ORCHESTRATOR.signal, "signal", fake_signal)
    monkeypatch.setattr(
        ORCHESTRATOR, "create_bucket", interrupt_create_bucket
    )
    monkeypatch.setattr(
        ORCHESTRATOR, "cleanup", lambda *_args: cleanup_calls.append(True)
    )

    with pytest.raises(KeyboardInterrupt):
        ORCHESTRATOR.execute(args)
    persisted = json.loads(args.state_file.read_text())
    failure = [
        event
        for event in persisted["events"]
        if event["kind"] == "execution_failure"
    ][-1]["value"]
    assert failure["interrupted"]
    assert cleanup_calls == [True]


def test_execute_direct_keyboard_interrupt_is_recorded(monkeypatch, tmp_path):
    args = SimpleNamespace(
        state_file=tmp_path / "state.json",
        task="0728T075",
        run_id="run-test",
        region="ap-northeast-1",
    )
    cleanup_calls = []

    def interrupt_create_bucket(*_args):
        raise KeyboardInterrupt

    monkeypatch.setattr(
        ORCHESTRATOR, "create_bucket", interrupt_create_bucket
    )
    monkeypatch.setattr(
        ORCHESTRATOR, "cleanup", lambda *_args: cleanup_calls.append(True)
    )

    with pytest.raises(KeyboardInterrupt):
        ORCHESTRATOR.execute(args)
    persisted = json.loads(args.state_file.read_text())
    failure = [
        event
        for event in persisted["events"]
        if event["kind"] == "execution_failure"
    ][-1]["value"]
    assert failure["interrupted"]
    assert cleanup_calls == [True]


def test_parse_args_requires_explicit_task(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["ec2_hunt_orchestrator.py"])
    with pytest.raises(SystemExit):
        ORCHESTRATOR.parse_args()
    assert "--task" in capsys.readouterr().err


def test_source_archive_is_deterministic(tmp_path):
    repo_root = tmp_path / "repo"
    probe = repo_root / "latency-probe"
    (probe / "src").mkdir(parents=True)
    (probe / "Cargo.toml").write_text("[package]\nname='probe'\n")
    (probe / "src" / "lib.rs").write_text("pub fn value() -> u8 { 1 }\n")
    (probe / "src" / "main.rs").write_text("fn main() {}\n")
    linux_lock = tmp_path / "Cargo.lock"
    linux_lock.write_text("version = 4\n")
    first = tmp_path / "first.tar.gz"
    second = tmp_path / "second.tar.gz"

    BUILD.create_source_archive(repo_root, linux_lock, first)
    BUILD.create_source_archive(repo_root, linux_lock, second)

    assert BUILD.sha256(first) == BUILD.sha256(second)


def test_setup_stdout_is_reserved_for_clock_receipt():
    script = Path(__file__).with_name("ec2_hunt_remote.sh").read_text()
    assert 'apt-get update >> "$setup_log" 2>&1' in script
    assert 'cat /home/admin/latency-hunt/setup-clock.json' in script
    assert script.index('cat /home/admin/latency-hunt/setup-clock.json') < (
        script.index('return "$clock_status"')
    )


def test_remote_upload_failure_returns_nonzero_receipt(tmp_path):
    archive = tmp_path / "smoke.tar.gz"
    archive.write_bytes(b"archive")
    script = Path(__file__).with_name("ec2_hunt_remote.sh")
    shell = r'''
curl() { return 22; }
sha256sum() { printf 'deadbeef  %s\n' "$1"; }
stat() { printf '7\n'; }
jq() {
  printf '{"label":"smoke","instance_id":"i-test","archive_sha256":"deadbeef","archive_bytes":7,"upload_status":1}\n'
}
source "$1"
upload_archive "$2" "https://example.invalid/upload" "smoke" "i-test"
'''
    result = subprocess.run(
        ["bash", "-c", shell, "bash", str(script), str(archive)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 1
    assert json.loads(result.stdout)["upload_status"] == 1


def test_setup_clock_receipt_is_a_success_gate():
    valid = {
        "captured_at_utc": "2026-07-28T10:10:03Z",
        "leap_status": "Normal",
        "error_bound_us": 200.0,
        "error_bound_limit_us": 750.0,
        "gate": "pass",
    }
    assert ORCHESTRATOR.parse_setup_clock_receipt(
        json.dumps(valid)
    ) == valid
    with pytest.raises(RuntimeError, match="not valid JSON"):
        ORCHESTRATOR.parse_setup_clock_receipt('{"truncated":')
    with pytest.raises(RuntimeError, match="failed gate"):
        ORCHESTRATOR.parse_setup_clock_receipt(
            json.dumps({**valid, "gate": "fail"})
        )


def test_failed_setup_command_retains_clock_receipt(
    monkeypatch, tmp_path
):
    state_path = tmp_path / "state.json"
    state = ORCHESTRATOR.load_state(
        state_path, "0728T074", "run-test", "ap-northeast-1"
    )
    details = {
        "i-test": {
            "Status": "Failed",
            "StandardOutputContent": json.dumps(
                {
                    "captured_at_utc": "2026-07-28T10:10:03Z",
                    "leap_status": "Normal",
                    "error_bound_us": 800.0,
                    "error_bound_limit_us": 750.0,
                    "gate": "fail",
                }
            ),
        }
    }

    def fail_wait(*_args):
        raise ORCHESTRATOR.CommandExecutionError("failed", details)

    monkeypatch.setattr(ORCHESTRATOR, "wait_command", fail_wait)
    with pytest.raises(
        ORCHESTRATOR.CommandExecutionError, match="failed"
    ):
        ORCHESTRATOR.wait_and_record_command(
            "ap-northeast-1",
            state_path,
            state,
            "command-test",
            ["i-test"],
            10,
        )
    persisted = json.loads(state_path.read_text())
    event = persisted["events"][-1]
    assert event["kind"] == "command_failure_details"
    assert "800.0" in event["value"]["details"]["i-test"][
        "StandardOutputContent"
    ]
