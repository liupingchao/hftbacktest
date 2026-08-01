#!/usr/bin/env python3
import argparse
import csv
import json
import signal
import shlex
import subprocess
import time
from pathlib import Path

import ec2_hunt
import ec2_hunt_orchestrator as control


CANDIDATE_COUNT = 7
INSTANCE_TYPE = "c6in.xlarge"
DURATION_SECONDS = 900
WARMUP_MESSAGES = 100


def create_spread_group(args, state_path, state):
    group = f"hft-{args.run_id[-20:]}-c6in-spread"
    control.aws(
        args.region,
        "ec2",
        "create-placement-group",
        "--group-name",
        group,
        "--strategy",
        "spread",
        "--spread-level",
        "rack",
        "--tag-specifications",
        control.tag_spec(
            "placement-group",
            args.task,
            args.run_id,
            "candidate-spread",
            group,
        ),
    )
    control.record(state_path, state, "placement_groups", group)
    return group


def launch_batch(args, state_path, state, group, candidate_indexes):
    candidates = []
    for index in candidate_indexes:
        label = f"candidate-{index:02d}"
        name = f"{args.task}-c6in-{index:02d}"
        try:
            result = control.aws(
                args.region,
                "ec2",
                "run-instances",
                "--image-id",
                args.image_id,
                "--instance-type",
                INSTANCE_TYPE,
                "--count",
                "1",
                "--subnet-id",
                args.subnet_id,
                "--security-group-ids",
                args.security_group_id,
                "--iam-instance-profile",
                f"Name={args.iam_instance_profile}",
                "--key-name",
                args.key_name,
                "--placement",
                f"AvailabilityZone={args.availability_zone},GroupName={group}",
                "--user-data",
                f"file://{args.user_data_file}",
                "--tag-specifications",
                control.tag_spec(
                    "instance",
                    args.task,
                    args.run_id,
                    "candidate",
                    name,
                ),
                "--query",
                "Instances[0].InstanceId",
                "--output",
                "text",
            )
            instance_id = result.stdout.strip()
            control.record(state_path, state, "instances", instance_id)
            control.record(
                state_path,
                state,
                "candidate_launched",
                {
                    "candidate_label": label,
                    "instance_id": instance_id,
                    "instance_type": INSTANCE_TYPE,
                    "placement_group": group,
                },
            )
            candidates.append((INSTANCE_TYPE, instance_id, group, label))
        except BaseException as error:
            control.record(
                state_path,
                state,
                "launch_failure",
                {"candidate_label": label, "error": str(error)},
            )
            raise
    return candidates


def protected_instance_receipt(args):
    result = control.aws(
        args.region,
        "ec2",
        "describe-instances",
        "--instance-ids",
        args.protected_instance_id,
        "--query",
        (
            "Reservations[0].Instances[0].{"
            "InstanceId:InstanceId,InstanceType:InstanceType,"
            "State:State.Name,AvailabilityZone:Placement.AvailabilityZone,"
            "LaunchTime:LaunchTime,PrivateIpAddress:PrivateIpAddress}"
        ),
        "--output",
        "json",
    )
    receipt = json.loads(result.stdout)
    if (
        receipt.get("InstanceId") != args.protected_instance_id
        or receipt.get("State") != "running"
    ):
        raise RuntimeError(f"protected instance is not running: {receipt}")
    return receipt


def write_candidates(run_dir, candidates):
    (run_dir / "candidate_instances.tsv").write_text(
        "".join(
            f"{kind}\t{instance_id}\t{group}\t{label}\n"
            for kind, instance_id, group, label in candidates
        )
    )


def capture_placement_receipt(args, state_path, state, group, candidates):
    instance_ids = [candidate[1] for candidate in candidates]
    group_result = control.aws(
        args.region,
        "ec2",
        "describe-placement-groups",
        "--group-names",
        group,
        "--query",
        (
            "PlacementGroups[0].{GroupName:GroupName,State:State,"
            "Strategy:Strategy,SpreadLevel:SpreadLevel,GroupArn:GroupArn}"
        ),
        "--output",
        "json",
    )
    instances_result = control.aws(
        args.region,
        "ec2",
        "describe-instances",
        "--instance-ids",
        *instance_ids,
        "--query",
        (
            "Reservations[].Instances[].{InstanceId:InstanceId,"
            "InstanceType:InstanceType,State:State.Name,"
            "AvailabilityZone:Placement.AvailabilityZone,"
            "GroupName:Placement.GroupName,PrivateIpAddress:PrivateIpAddress,"
            "LaunchTime:LaunchTime}"
        ),
        "--output",
        "json",
    )
    group_receipt = json.loads(group_result.stdout)
    instance_receipts = sorted(
        json.loads(instances_result.stdout),
        key=lambda row: row["InstanceId"],
    )
    verified = (
        group_receipt.get("GroupName") == group
        and group_receipt.get("State") == "available"
        and group_receipt.get("Strategy") == "spread"
        and str(group_receipt.get("SpreadLevel", "")).lower() == "rack"
        and len(instance_receipts) == CANDIDATE_COUNT
        and all(
            row["InstanceType"] == INSTANCE_TYPE
            and row["AvailabilityZone"] == args.availability_zone
            and row["GroupName"] == group
            and row["State"] == "running"
            for row in instance_receipts
        )
    )
    receipt = {
        "verified": verified,
        "placement_group": group_receipt,
        "instances": instance_receipts,
    }
    control.record(state_path, state, "placement_receipt", receipt)
    if not verified:
        raise RuntimeError(f"placement receipt failed: {receipt}")
    return receipt


def setup_hosts(args, state_path, state, instance_ids, setup_command):
    control.aws(
        args.region,
        "ec2",
        "wait",
        "instance-status-ok",
        "--instance-ids",
        *instance_ids,
    )
    control.wait_for_ssm(args.region, instance_ids)
    command_id = control.send_command(
        args.region, instance_ids, setup_command, 1800
    )
    control.record(state_path, state, "commands", command_id)
    details = control.wait_and_record_command(
        args.region,
        state_path,
        state,
        command_id,
        instance_ids,
        2100,
    )
    control.verify_setup_clock_receipts(
        state_path, state, command_id, details
    )


def run_window(args, state_path, state, instance_ids, label, duration, warmup):
    commands = {
        instance_id: (
            "/tmp/ec2_hunt_remote.sh run "
            f"{shlex.quote(control.presigned_url(args.region, args.bucket, f'{args.task}/results/{instance_id}/{label}.tar.gz', 'put_object'))} "
            f"{duration} {warmup} {label}"
        )
        for instance_id in instance_ids
    }
    details = control.run_commands_per_instance(
        args.region, state_path, state, commands, duration + 900
    )
    receipts = control.verify_uploaded_archives(
        args, state_path, state, label, details
    )
    return control.download_and_verify_archives(
        args, state_path, state, label, receipts
    )


def write_report(report, json_output, csv_output):
    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    fields = [
        "candidate_label",
        "instance_type",
        "instance_id",
        "eligible",
        "feed_eligible",
        "local_pipeline_eligible",
        "clock_before_bound_us",
        "clock_after_bound_us",
        "clock_max_bound_us",
        *ec2_hunt.METRICS,
    ]
    with csv_output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in report["rows"]:
            writer.writerow({field: row[field] for field in fields})


def local_rebuild(args, report):
    output_dir = args.run_dir / "local_rebuild"
    output_dir.mkdir(parents=True, exist_ok=True)
    receipts = []
    for row in report["rows"]:
        result_dir = (
            args.run_dir / "extracted" / row["instance_id"] / "full"
        )
        for venue in ("binance", "hyperliquid"):
            output = output_dir / f"{row['instance_id']}-{venue}.json"
            subprocess.run(
                [
                    str(args.local_probe),
                    "summarize",
                    "--raw-input",
                    str(result_dir / f"{venue}.ndjson"),
                    "--summary-output",
                    str(output),
                ],
                check=True,
            )
            expected = result_dir / f"{venue}-summary.json"
            matched = output.read_bytes() == expected.read_bytes()
            receipts.append(
                {
                    "instance_id": row["instance_id"],
                    "venue": venue,
                    "matched": matched,
                    "rebuilt_sha256": control.sha256(output),
                    "expected_sha256": control.sha256(expected),
                }
            )
            if not matched:
                raise RuntimeError(
                    f"local rebuild mismatch: {row['instance_id']} {venue}"
                )
    return receipts


def disable_termination_protection(region, instance_ids):
    for instance_id in instance_ids:
        control.aws(
            region,
            "ec2",
            "modify-instance-attribute",
            "--instance-id",
            instance_id,
            "--disable-api-termination",
            "Value=false",
            check=False,
        )


def empty_and_delete_bucket(args):
    if not control.bucket_exists(args.region, args.bucket):
        return
    result = control.aws(
        args.region,
        "s3",
        "rm",
        f"s3://{args.bucket}",
        "--recursive",
        check=False,
    )
    if result.returncode:
        raise RuntimeError(f"empty bucket failed: {result.stderr.strip()}")
    result = control.aws(
        args.region,
        "s3api",
        "delete-bucket",
        "--bucket",
        args.bucket,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(f"delete bucket failed: {result.stderr.strip()}")


def finalize_winner(
    args, state_path, state, candidates, group, report, protected_before
):
    winner = report["balanced_selection"]["winner_instance_id"]
    if report["candidate_count"] != CANDIDATE_COUNT:
        raise RuntimeError("candidate count is not seven")
    if report["eligible_count"] != CANDIDATE_COUNT:
        raise RuntimeError("not all seven candidates are eligible")
    instance_ids = [candidate[1] for candidate in candidates]
    if winner not in instance_ids:
        raise RuntimeError(f"selected winner is outside cohort: {winner}")
    losers = sorted(set(instance_ids) - {winner})

    control.aws(
        args.region,
        "ec2",
        "create-tags",
        "--resources",
        winner,
        "--tags",
        "Key=HuntDisposition,Value=winner",
        f"Key=RetainedByTask,Value={args.task}",
        f"Key=Name,Value={args.task}-c6in-winner",
    )
    control.aws(
        args.region,
        "ec2",
        "modify-instance-attribute",
        "--instance-id",
        winner,
        "--disable-api-termination",
        "Value=true",
    )
    protection_before_result = control.aws(
        args.region,
        "ec2",
        "describe-instance-attribute",
        "--instance-id",
        winner,
        "--attribute",
        "disableApiTermination",
        "--output",
        "json",
    )
    protection_before = json.loads(protection_before_result.stdout)
    if (
        protection_before.get("DisableApiTermination", {}).get("Value")
        is not True
    ):
        raise RuntimeError(
            f"winner protection did not become active: {protection_before}"
        )
    control.record(
        state_path,
        state,
        "winner_protected_before_loser_cleanup",
        {
            "winner_instance_id": winner,
            "termination_protection": protection_before,
            "loser_instance_ids": losers,
        },
    )

    control.aws(
        args.region,
        "ec2",
        "terminate-instances",
        "--instance-ids",
        *losers,
    )
    control.aws(
        args.region,
        "ec2",
        "wait",
        "instance-terminated",
        "--instance-ids",
        *losers,
    )
    empty_and_delete_bucket(args)

    winner_result = control.aws(
        args.region,
        "ec2",
        "describe-instances",
        "--instance-ids",
        winner,
        "--query",
        (
            "Reservations[0].Instances[0].{"
            "InstanceId:InstanceId,InstanceType:InstanceType,"
            "State:State.Name,AvailabilityZone:Placement.AvailabilityZone,"
            "GroupName:Placement.GroupName,LaunchTime:LaunchTime,"
            "PrivateIpAddress:PrivateIpAddress,Tags:Tags}"
        ),
        "--output",
        "json",
    )
    winner_receipt = json.loads(winner_result.stdout)
    protection_after_result = control.aws(
        args.region,
        "ec2",
        "describe-instance-attribute",
        "--instance-id",
        winner,
        "--attribute",
        "disableApiTermination",
        "--output",
        "json",
    )
    protection_after = json.loads(protection_after_result.stdout)
    protected_after = protected_instance_receipt(args)
    active_task_instances = control.discover_instance_ids(
        args.region, args.task, args.run_id
    )
    groups = control.discover_placement_groups(
        args.region, args.task, args.run_id
    )
    buckets = control.discover_buckets(args.region, args.task, args.run_id)
    verified = (
        winner_receipt.get("State") == "running"
        and winner_receipt.get("InstanceType") == INSTANCE_TYPE
        and winner_receipt.get("GroupName") == group
        and protection_before.get("DisableApiTermination", {}).get("Value")
        is True
        and protection_after.get("DisableApiTermination", {}).get("Value")
        is True
        and active_task_instances == [winner]
        and groups == [group]
        and not buckets
        and protected_before["InstanceId"] == protected_after["InstanceId"]
        and protected_after["State"] == "running"
    )
    final_receipt = {
        "verified": verified,
        "winner": winner_receipt,
        "winner_termination_protection_before_cleanup": protection_before,
        "winner_termination_protection": protection_after,
        "loser_instance_ids": losers,
        "active_task_instances": active_task_instances,
        "placement_groups": groups,
        "buckets": buckets,
        "protected_instance_before": protected_before,
        "protected_instance_after": protected_after,
    }
    control.record(state_path, state, "retained_winner", final_receipt)
    if not verified:
        raise RuntimeError(f"winner finalization verification failed: {final_receipt}")
    return final_receipt


def failure_cleanup(state_path, state):
    disable_termination_protection(state["region"], state.get("instances", []))
    return control.cleanup(state_path, state)


def execute(args):
    if args.task != "0729T001":
        raise RuntimeError("this controller is frozen to task 0729T001")
    state_path = args.state_file
    state = control.load_state(
        state_path, args.task, args.run_id, args.region
    )
    interrupted = False
    completed = False

    def handle_signal(_signum, _frame):
        nonlocal interrupted
        interrupted = True
        raise KeyboardInterrupt

    previous_int = signal.signal(signal.SIGINT, handle_signal)
    previous_term = signal.signal(signal.SIGTERM, handle_signal)
    try:
        protected_before = protected_instance_receipt(args)
        control.record(
            state_path, state, "protected_instance_preflight", protected_before
        )
        control.create_bucket(args, state_path, state)
        control.upload_artifacts(args)
        args.run_dir.mkdir(parents=True, exist_ok=True)
        (args.run_dir / "build").mkdir(parents=True, exist_ok=True)
        (args.run_dir / "build" / "artifact-sha256.txt").write_text(
            f"{args.binary_sha256}  latency-probe-linux-x86_64\n"
        )

        group = create_spread_group(args, state_path, state)
        candidates = launch_batch(
            args, state_path, state, group, range(1, 3)
        )
        canary_ids = [candidate[1] for candidate in candidates]
        script_url = control.presigned_url(
            args.region,
            args.bucket,
            f"{args.task}/ec2_hunt_remote.sh",
            "get_object",
        )
        binary_url = control.presigned_url(
            args.region,
            args.bucket,
            f"{args.task}/latency-probe-linux-x86_64",
            "get_object",
        )
        remote = (
            f"curl -fsS {shlex.quote(script_url)} -o /tmp/ec2_hunt_remote.sh "
            "&& chmod 0755 /tmp/ec2_hunt_remote.sh"
        )
        setup_command = (
            f"{remote} && /tmp/ec2_hunt_remote.sh setup "
            f"{shlex.quote(binary_url)} {args.binary_sha256}"
        )
        setup_hosts(
            args, state_path, state, canary_ids, setup_command
        )
        run_window(
            args,
            state_path,
            state,
            canary_ids,
            "smoke",
            args.smoke_duration_seconds,
            10,
        )

        remaining = launch_batch(
            args, state_path, state, group, range(3, CANDIDATE_COUNT + 1)
        )
        candidates.extend(remaining)
        setup_hosts(
            args,
            state_path,
            state,
            [candidate[1] for candidate in remaining],
            setup_command,
        )
        capture_placement_receipt(
            args, state_path, state, group, candidates
        )
        write_candidates(args.run_dir, candidates)
        run_window(
            args,
            state_path,
            state,
            [candidate[1] for candidate in candidates],
            "full",
            DURATION_SECONDS,
            WARMUP_MESSAGES,
        )

        report = ec2_hunt.analyze(args.run_dir)
        rebuild_receipts = local_rebuild(args, report)
        report["local_rebuild"] = {
            "matched_count": sum(
                receipt["matched"] for receipt in rebuild_receipts
            ),
            "expected_count": CANDIDATE_COUNT * 2,
            "receipts": rebuild_receipts,
        }
        if report["local_rebuild"]["matched_count"] != CANDIDATE_COUNT * 2:
            raise RuntimeError("local rebuild count mismatch")
        write_report(report, args.json_output, args.csv_output)
        final_receipt = finalize_winner(
            args,
            state_path,
            state,
            candidates,
            group,
            report,
            protected_before,
        )
        (args.run_dir / "final_receipt.json").write_text(
            json.dumps(final_receipt, indent=2, sort_keys=True) + "\n"
        )
        control.record(state_path, state, "execution", "success")
        completed = True
    except BaseException as error:
        if isinstance(error, KeyboardInterrupt):
            interrupted = True
        control.record(
            state_path,
            state,
            "execution_failure",
            {"interrupted": interrupted, "error": str(error)},
        )
        raise
    finally:
        try:
            if not completed:
                failure_cleanup(state_path, state)
        finally:
            signal.signal(signal.SIGINT, previous_int)
            signal.signal(signal.SIGTERM, previous_term)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a seven-host c6in spread hunt and retain its winner"
    )
    parser.add_argument("--task", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--region", default="ap-northeast-1")
    parser.add_argument("--availability-zone", default="ap-northeast-1c")
    parser.add_argument("--image-id", required=True)
    parser.add_argument("--subnet-id", required=True)
    parser.add_argument("--security-group-id", required=True)
    parser.add_argument("--iam-instance-profile", required=True)
    parser.add_argument("--key-name", default="key1")
    parser.add_argument("--user-data-file", type=Path, required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--binary-sha256", required=True)
    parser.add_argument("--local-probe", type=Path, required=True)
    parser.add_argument("--remote-script", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--state-file", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path, required=True)
    parser.add_argument("--protected-instance-id", required=True)
    parser.add_argument("--smoke-duration-seconds", type=int, default=60)
    return parser.parse_args()


if __name__ == "__main__":
    execute(parse_args())
