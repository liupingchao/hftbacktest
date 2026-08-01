#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import signal
import shlex
import subprocess
import tempfile
import time
from pathlib import Path

import botocore.session
from botocore.config import Config


TERMINAL_COMMAND_STATUSES = {
    "Success",
    "Cancelled",
    "Failed",
    "TimedOut",
    "Cancelling",
}


class CommandExecutionError(RuntimeError):
    def __init__(self, message, details):
        super().__init__(message)
        self.details = details


def atomic_write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, delete=False, prefix=f".{path.name}."
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temp_name = handle.name
    os.replace(temp_name, path)


def load_state(path, task, run_id, region):
    if path.exists():
        return json.loads(path.read_text())
    return {
        "schema_version": "ec2-latency-hunt-state-v1",
        "task": task,
        "run_id": run_id,
        "region": region,
        "instances": [],
        "placement_groups": [],
        "buckets": [],
        "commands": [],
        "events": [],
    }


def record(state_path, state, kind, value):
    if kind in {"instances", "placement_groups", "buckets", "commands"}:
        if value not in state[kind]:
            state[kind].append(value)
    state["events"].append(
        {
            "at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "kind": kind,
            "value": value,
        }
    )
    atomic_write_json(state_path, state)


def aws(region, *args, check=True):
    command = ["aws", *args, "--region", region]
    result = subprocess.run(command, check=False, text=True, capture_output=True)
    if check and result.returncode:
        raise RuntimeError(
            f"AWS command failed ({result.returncode}): {' '.join(command)}\n"
            f"{result.stderr.strip()}"
        )
    return result


def error_has_any(result, markers):
    message = f"{result.stdout}\n{result.stderr}".lower()
    return any(marker.lower() in message for marker in markers)


def bucket_exists(region, bucket):
    result = aws(
        region, "s3api", "head-bucket", "--bucket", bucket, check=False
    )
    if result.returncode == 0:
        return True
    if error_has_any(
        result, ("404", "Not Found", "NoSuchBucket", "NotFound")
    ):
        return False
    raise RuntimeError(
        f"unable to determine whether bucket {bucket} exists: "
        f"{result.stderr.strip()}"
    )


def discover_instance_ids(region, task, run_id):
    result = aws(
        region,
        "ec2",
        "describe-instances",
        "--filters",
        f"Name=tag:Task,Values={task}",
        f"Name=tag:RunId,Values={run_id}",
        "Name=instance-state-name,Values=pending,running,stopping,stopped",
        "--query",
        "Reservations[].Instances[].InstanceId",
        "--output",
        "json",
    )
    return json.loads(result.stdout)


def discover_placement_groups(region, task, run_id):
    result = aws(
        region,
        "ec2",
        "describe-placement-groups",
        "--filters",
        f"Name=tag:Task,Values={task}",
        f"Name=tag:RunId,Values={run_id}",
        "--query",
        "PlacementGroups[].GroupName",
        "--output",
        "json",
    )
    return json.loads(result.stdout)


def discover_buckets(region, task, run_id):
    result = aws(
        region,
        "s3api",
        "list-buckets",
        "--query",
        "Buckets[].Name",
        "--output",
        "json",
    )
    matches = []
    for bucket in json.loads(result.stdout):
        tags_result = aws(
            region,
            "s3api",
            "get-bucket-tagging",
            "--bucket",
            bucket,
            "--output",
            "json",
            check=False,
        )
        if tags_result.returncode:
            if error_has_any(
                tags_result,
                ("NoSuchTagSet", "NoSuchBucket", "404", "Not Found"),
            ):
                continue
            raise RuntimeError(
                f"unable to inspect tags for bucket {bucket}: "
                f"{tags_result.stderr.strip()}"
            )
        tags = {
            row["Key"]: row["Value"]
            for row in json.loads(tags_result.stdout).get("TagSet", [])
        }
        if tags.get("Task") == task and tags.get("RunId") == run_id:
            matches.append(bucket)
    return matches


def cleanup_plan(
    state,
    discovered_instances=(),
    discovered_groups=(),
    discovered_buckets=(),
):
    return {
        "instances": sorted(set(state.get("instances", ())) | set(discovered_instances)),
        "placement_groups": sorted(
            set(state.get("placement_groups", ())) | set(discovered_groups)
        ),
        "buckets": sorted(
            set(state.get("buckets", ())) | set(discovered_buckets)
        ),
    }


def cleanup(state_path, state):
    region = state["region"]
    discovery_errors = []
    try:
        discovered_instances = discover_instance_ids(
            region, state["task"], state["run_id"]
        )
    except BaseException as error:
        discovered_instances = []
        discovery_errors.append(f"instance discovery: {error}")
    try:
        discovered_groups = discover_placement_groups(
            region, state["task"], state["run_id"]
        )
    except BaseException as error:
        discovered_groups = []
        discovery_errors.append(f"placement group discovery: {error}")
    try:
        discovered_buckets = discover_buckets(
            region, state["task"], state["run_id"]
        )
    except BaseException as error:
        discovered_buckets = []
        discovery_errors.append(f"bucket discovery: {error}")
    plan = cleanup_plan(
        state,
        discovered_instances,
        discovered_groups,
        discovered_buckets,
    )
    errors = list(discovery_errors)
    if plan["instances"]:
        result = aws(
            region,
            "ec2",
            "terminate-instances",
            "--instance-ids",
            *plan["instances"],
            check=False,
        )
        if result.returncode:
            errors.append(f"terminate instances: {result.stderr.strip()}")
        else:
            result = aws(
                region,
                "ec2",
                "wait",
                "instance-terminated",
                "--instance-ids",
                *plan["instances"],
                check=False,
            )
            if result.returncode:
                errors.append(f"wait instance terminated: {result.stderr.strip()}")
    for group in plan["placement_groups"]:
        result = aws(
            region,
            "ec2",
            "delete-placement-group",
            "--group-name",
            group,
            check=False,
        )
        if result.returncode and "InvalidPlacementGroup.Unknown" not in result.stderr:
            errors.append(
                f"delete placement group {group}: {result.stderr.strip()}"
            )
    for bucket in plan["buckets"]:
        try:
            exists = bucket_exists(region, bucket)
        except BaseException as error:
            errors.append(str(error))
            continue
        if not exists:
            continue
        result = aws(
            region, "s3", "rm", f"s3://{bucket}", "--recursive", check=False
        )
        if result.returncode:
            errors.append(f"empty bucket {bucket}: {result.stderr.strip()}")
        result = aws(
            region, "s3api", "delete-bucket", "--bucket", bucket, check=False
        )
        if result.returncode:
            errors.append(f"delete bucket {bucket}: {result.stderr.strip()}")

    post_state = {
        "instances": [],
        "placement_groups": [],
        "buckets": [],
    }
    try:
        post_state["instances"] = discover_instance_ids(
            region, state["task"], state["run_id"]
        )
    except BaseException as error:
        errors.append(f"post-cleanup instance discovery: {error}")
    try:
        post_state["placement_groups"] = discover_placement_groups(
            region, state["task"], state["run_id"]
        )
    except BaseException as error:
        errors.append(f"post-cleanup placement group discovery: {error}")
    try:
        post_state["buckets"] = discover_buckets(
            region, state["task"], state["run_id"]
        )
    except BaseException as error:
        errors.append(f"post-cleanup bucket discovery: {error}")
    for bucket in plan["buckets"]:
        if bucket in post_state["buckets"]:
            continue
        try:
            if bucket_exists(region, bucket):
                post_state["buckets"].append(bucket)
        except BaseException as error:
            errors.append(f"post-cleanup bucket check {bucket}: {error}")

    verified = not errors and not any(post_state.values())
    result = {
        "plan": plan,
        "errors": errors,
        "post_state": post_state,
        "verified": verified,
    }
    record(state_path, state, "cleanup", result)
    if not verified:
        record(state_path, state, "cleanup_failure", result)
        raise RuntimeError(f"cleanup verification failed: {result}")
    return result


def tag_spec(resource_type, task, run_id, role, name):
    return json.dumps(
        [
            {
                "ResourceType": resource_type,
                "Tags": [
                    {"Key": "Task", "Value": task},
                    {"Key": "RunId", "Value": run_id},
                    {"Key": "Role", "Value": role},
                    {"Key": "Name", "Value": name},
                ],
            }
        ]
    )


def launch_candidates(args, state_path, state, instance_types):
    candidates = []
    for instance_type in instance_types:
        group = f"hft-{args.run_id[-20:]}-{instance_type.replace('.', '-')}"
        name = f"{args.task}-{instance_type}"
        try:
            aws(
                args.region,
                "ec2",
                "create-placement-group",
                "--group-name",
                group,
                "--strategy",
                "cluster",
                "--tag-specifications",
                tag_spec("placement-group", args.task, args.run_id, "candidate", name),
            )
            record(state_path, state, "placement_groups", group)
            result = aws(
                args.region,
                "ec2",
                "run-instances",
                "--image-id",
                args.image_id,
                "--instance-type",
                instance_type,
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
                tag_spec("instance", args.task, args.run_id, "candidate", name),
                "--query",
                "Instances[0].InstanceId",
                "--output",
                "text",
            )
            instance_id = result.stdout.strip()
            record(state_path, state, "instances", instance_id)
            candidates.append((instance_type, instance_id, group))
        except BaseException as error:
            record(
                state_path,
                state,
                "launch_failure",
                {"instance_type": instance_type, "error": str(error)},
            )
            raise
    return candidates


def wait_for_ssm(region, instance_ids, timeout_seconds=900):
    deadline = time.monotonic() + timeout_seconds
    pending = set(instance_ids)
    while pending and time.monotonic() < deadline:
        result = aws(
            region,
            "ssm",
            "describe-instance-information",
            "--filters",
            f"Key=InstanceIds,Values={','.join(sorted(pending))}",
            "--query",
            "InstanceInformationList[?PingStatus==`Online`].InstanceId",
            "--output",
            "json",
        )
        pending -= set(json.loads(result.stdout))
        if pending:
            time.sleep(10)
    if pending:
        raise RuntimeError(f"SSM registration timeout: {sorted(pending)}")


def send_command(region, instance_ids, command, timeout_seconds):
    parameters = json.dumps(
        {"commands": [command], "executionTimeout": [str(timeout_seconds)]}
    )
    result = aws(
        region,
        "ssm",
        "send-command",
        "--instance-ids",
        *instance_ids,
        "--document-name",
        "AWS-RunShellScript",
        "--parameters",
        parameters,
        "--timeout-seconds",
        str(timeout_seconds),
        "--query",
        "Command.CommandId",
        "--output",
        "text",
    )
    return result.stdout.strip()


def wait_command(region, command_id, instance_ids, timeout_seconds):
    deadline = time.monotonic() + timeout_seconds
    final = {}
    while time.monotonic() < deadline:
        result = aws(
            region,
            "ssm",
            "list-command-invocations",
            "--command-id",
            command_id,
            "--details",
            "--query",
            "CommandInvocations[].{InstanceId:InstanceId,Status:Status}",
            "--output",
            "json",
        )
        rows = json.loads(result.stdout)
        final = {row["InstanceId"]: row["Status"] for row in rows}
        if len(final) == len(instance_ids) and all(
            status in TERMINAL_COMMAND_STATUSES for status in final.values()
        ):
            break
        time.sleep(10)
    if set(final) != set(instance_ids) or any(
        status != "Success" for status in final.values()
    ):
        details = command_details(region, command_id, instance_ids)
        raise CommandExecutionError(
            f"SSM command {command_id} failed: {final}", details
        )
    return command_details(region, command_id, instance_ids)


def command_details(region, command_id, instance_ids):
    details = {}
    for instance_id in instance_ids:
        result = aws(
            region,
            "ssm",
            "get-command-invocation",
            "--command-id",
            command_id,
            "--instance-id",
            instance_id,
            "--output",
            "json",
            check=False,
        )
        if result.returncode:
            details[instance_id] = {
                "Status": "InvocationLookupFailed",
                "StandardErrorContent": result.stderr.strip(),
            }
            continue
        details[instance_id] = json.loads(result.stdout)
    return details


def wait_and_record_command(
    region,
    state_path,
    state,
    command_id,
    instance_ids,
    timeout_seconds,
):
    try:
        return wait_command(
            region, command_id, instance_ids, timeout_seconds
        )
    except CommandExecutionError as error:
        record(
            state_path,
            state,
            "command_failure_details",
            {"command_id": command_id, "details": error.details},
        )
        raise


def create_bucket(args, state_path, state):
    record(state_path, state, "buckets", args.bucket)
    record(state_path, state, "bucket_planned", args.bucket)
    aws(
        args.region,
        "s3api",
        "create-bucket",
        "--bucket",
        args.bucket,
        "--create-bucket-configuration",
        f"LocationConstraint={args.region}",
    )
    aws(
        args.region,
        "s3api",
        "put-bucket-tagging",
        "--bucket",
        args.bucket,
        "--tagging",
        json.dumps(
            {
                "TagSet": [
                    {"Key": "Task", "Value": args.task},
                    {"Key": "RunId", "Value": args.run_id},
                ]
            }
        ),
    )
    record(state_path, state, "bucket_created", args.bucket)
    aws(
        args.region,
        "s3api",
        "put-public-access-block",
        "--bucket",
        args.bucket,
        "--public-access-block-configuration",
        "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true",
    )


def upload_artifacts(args):
    aws(
        args.region,
        "s3",
        "cp",
        str(args.binary),
        f"s3://{args.bucket}/{args.task}/latency-probe-linux-x86_64",
    )
    aws(
        args.region,
        "s3",
        "cp",
        str(args.remote_script),
        f"s3://{args.bucket}/{args.task}/ec2_hunt_remote.sh",
    )


def presigned_url(region, bucket, key, operation):
    client = botocore.session.get_session().create_client(
        "s3",
        region_name=region,
        endpoint_url=f"https://s3.{region}.amazonaws.com",
        config=Config(signature_version="s3v4", s3={"addressing_style": "virtual"}),
    )
    return client.generate_presigned_url(
        operation,
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=14_400,
    )


def run_commands_per_instance(
    region, state_path, state, commands_by_instance, timeout_seconds
):
    submitted = []
    for instance_id, command in commands_by_instance.items():
        command_id = send_command(region, [instance_id], command, timeout_seconds)
        record(state_path, state, "commands", command_id)
        submitted.append((command_id, instance_id))
    details = {}
    for command_id, instance_id in submitted:
        details.update(
            wait_and_record_command(
                region,
                state_path,
                state,
                command_id,
                [instance_id],
                timeout_seconds + 300,
            )
        )
    return details


def parse_upload_receipt(stdout):
    for line in reversed(stdout.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        required = {
            "label",
            "instance_id",
            "archive_sha256",
            "archive_bytes",
            "upload_status",
        }
        if isinstance(value, dict) and required <= value.keys():
            if value["upload_status"] != 0:
                raise RuntimeError(f"remote upload failed: {value}")
            if value["archive_bytes"] <= 0:
                raise RuntimeError(f"remote archive is empty: {value}")
            return value
    raise RuntimeError("remote upload receipt missing from SSM stdout")


def parse_setup_clock_receipt(stdout):
    try:
        receipt = json.loads(stdout.strip())
    except (json.JSONDecodeError, AttributeError) as error:
        raise RuntimeError(f"setup clock receipt is not valid JSON: {error}")
    required = {
        "captured_at_utc",
        "leap_status",
        "error_bound_us",
        "error_bound_limit_us",
        "gate",
    }
    if not isinstance(receipt, dict) or not required <= receipt.keys():
        raise RuntimeError(f"setup clock receipt fields missing: {receipt}")
    if (
        receipt["leap_status"] != "Normal"
        or receipt["gate"] != "pass"
        or receipt["error_bound_us"] is None
        or receipt["error_bound_us"] > receipt["error_bound_limit_us"]
    ):
        raise RuntimeError(f"setup clock receipt failed gate: {receipt}")
    return receipt


def verify_setup_clock_receipts(
    state_path, state, command_id, command_details_by_instance
):
    receipts = {}
    for instance_id, details in command_details_by_instance.items():
        receipt = parse_setup_clock_receipt(
            details.get("StandardOutputContent", "")
        )
        receipts[instance_id] = {
            "captured_at_utc": receipt["captured_at_utc"],
            "error_bound_us": receipt["error_bound_us"],
            "error_bound_limit_us": receipt["error_bound_limit_us"],
            "gate": receipt["gate"],
        }
    record(
        state_path,
        state,
        "setup_clock_receipts_verified",
        {"command_id": command_id, "receipts": receipts},
    )
    return receipts


def verify_uploaded_archives(
    args, state_path, state, label, command_details_by_instance
):
    receipts = {}
    for instance_id, details in command_details_by_instance.items():
        receipt = parse_upload_receipt(
            details.get("StandardOutputContent", "")
        )
        if receipt["instance_id"] != instance_id or receipt["label"] != label:
            raise RuntimeError(
                f"upload receipt identity mismatch for {instance_id}: {receipt}"
            )
        key = f"{args.task}/results/{instance_id}/{label}.tar.gz"
        result = aws(
            args.region,
            "s3api",
            "head-object",
            "--bucket",
            args.bucket,
            "--key",
            key,
            "--output",
            "json",
        )
        content_length = json.loads(result.stdout)["ContentLength"]
        if content_length != receipt["archive_bytes"]:
            raise RuntimeError(
                f"S3 size mismatch for {key}: "
                f"receipt={receipt['archive_bytes']} s3={content_length}"
            )
        receipts[instance_id] = {**receipt, "key": key}
    record(
        state_path,
        state,
        "upload_objects_verified",
        {"label": label, "receipts": receipts},
    )
    return receipts


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_and_verify_archives(
    args, state_path, state, label, receipts
):
    verified = {}
    for instance_id, receipt in receipts.items():
        output = (
            args.run_dir
            / f"{label}_archives"
            / instance_id
            / f"{label}.tar.gz"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        aws(
            args.region,
            "s3",
            "cp",
            f"s3://{args.bucket}/{receipt['key']}",
            str(output),
        )
        actual_sha = sha256(output)
        actual_bytes = output.stat().st_size
        if (
            actual_sha != receipt["archive_sha256"]
            or actual_bytes != receipt["archive_bytes"]
        ):
            raise RuntimeError(
                f"download verification failed for {instance_id}/{label}: "
                f"sha={actual_sha} bytes={actual_bytes}"
            )
        verified[instance_id] = {
            "path": str(output),
            "sha256": actual_sha,
            "bytes": actual_bytes,
        }
    record(
        state_path,
        state,
        "downloaded_archives_verified",
        {"label": label, "archives": verified},
    )
    return verified


def execute(args):
    state_path = args.state_file
    state = load_state(state_path, args.task, args.run_id, args.region)
    interrupted = False

    def handle_signal(_signum, _frame):
        nonlocal interrupted
        interrupted = True
        raise KeyboardInterrupt

    previous_int = signal.signal(signal.SIGINT, handle_signal)
    previous_term = signal.signal(signal.SIGTERM, handle_signal)
    try:
        create_bucket(args, state_path, state)
        upload_artifacts(args)
        args.run_dir.mkdir(parents=True, exist_ok=True)
        (args.run_dir / "build").mkdir(parents=True, exist_ok=True)
        (args.run_dir / "build" / "artifact-sha256.txt").write_text(
            f"{args.binary_sha256}  latency-probe-linux-x86_64\n"
        )
        instance_types = [
            value.strip() for value in args.instance_types.split(",") if value.strip()
        ]
        if len(instance_types) < 2:
            raise RuntimeError("at least two instance types are required for canary")
        candidates = launch_candidates(
            args, state_path, state, instance_types[:2]
        )
        canary_ids = [candidate[1] for candidate in candidates]
        aws(args.region, "ec2", "wait", "instance-status-ok", "--instance-ids", *canary_ids)
        wait_for_ssm(args.region, canary_ids)
        script_url = presigned_url(
            args.region, args.bucket, f"{args.task}/ec2_hunt_remote.sh", "get_object"
        )
        binary_url = presigned_url(
            args.region,
            args.bucket,
            f"{args.task}/latency-probe-linux-x86_64",
            "get_object",
        )
        remote = (
            f"curl -fsS {shlex.quote(script_url)} -o /tmp/ec2_hunt_remote.sh "
            "&& chmod 0755 /tmp/ec2_hunt_remote.sh"
        )
        setup = (
            f"{remote} && /tmp/ec2_hunt_remote.sh setup "
            f"{shlex.quote(binary_url)} {args.binary_sha256}"
        )
        command_id = send_command(args.region, canary_ids, setup, 1800)
        record(state_path, state, "commands", command_id)
        setup_details = wait_and_record_command(
            args.region,
            state_path,
            state,
            command_id,
            canary_ids,
            2100,
        )
        verify_setup_clock_receipts(
            state_path, state, command_id, setup_details
        )

        smoke_commands = {
            instance_id: (
                "/tmp/ec2_hunt_remote.sh run "
                f"{shlex.quote(presigned_url(args.region, args.bucket, f'{args.task}/results/{instance_id}/smoke.tar.gz', 'put_object'))} "
                f"{args.smoke_duration_seconds} 10 smoke"
            )
            for instance_id in canary_ids
        }
        smoke_details = run_commands_per_instance(
            args.region, state_path, state, smoke_commands, 900
        )
        smoke_receipts = verify_uploaded_archives(
            args, state_path, state, "smoke", smoke_details
        )
        download_and_verify_archives(
            args, state_path, state, "smoke", smoke_receipts
        )

        if args.smoke_only:
            (args.run_dir / "candidate_instances.tsv").write_text(
                "".join(
                    f"{kind}\t{iid}\t{group}\n"
                    for kind, iid, group in candidates
                )
            )
            record(state_path, state, "execution", "success")
            return

        remaining = launch_candidates(
            args, state_path, state, instance_types[2:]
        )
        candidates.extend(remaining)
        remaining_ids = [candidate[1] for candidate in remaining]
        if remaining_ids:
            aws(
                args.region,
                "ec2",
                "wait",
                "instance-status-ok",
                "--instance-ids",
                *remaining_ids,
            )
            wait_for_ssm(args.region, remaining_ids)
            command_id = send_command(args.region, remaining_ids, setup, 1800)
            record(state_path, state, "commands", command_id)
            setup_details = wait_and_record_command(
                args.region,
                state_path,
                state,
                command_id,
                remaining_ids,
                2100,
            )
            verify_setup_clock_receipts(
                state_path, state, command_id, setup_details
            )

        (args.run_dir / "candidate_instances.tsv").write_text(
            "".join(f"{kind}\t{iid}\t{group}\n" for kind, iid, group in candidates)
        )
        instance_ids = [candidate[1] for candidate in candidates]
        full_commands = {
            instance_id: (
                "/tmp/ec2_hunt_remote.sh run "
                f"{shlex.quote(presigned_url(args.region, args.bucket, f'{args.task}/results/{instance_id}/full.tar.gz', 'put_object'))} "
                "900 100 full"
            )
            for instance_id in instance_ids
        }
        full_details = run_commands_per_instance(
            args.region, state_path, state, full_commands, 1800
        )
        full_receipts = verify_uploaded_archives(
            args, state_path, state, "full", full_details
        )
        download_and_verify_archives(
            args, state_path, state, "full", full_receipts
        )
        record(state_path, state, "execution", "success")
    except BaseException as error:
        if isinstance(error, KeyboardInterrupt):
            interrupted = True
        record(
            state_path,
            state,
            "execution_failure",
            {"interrupted": interrupted, "error": str(error)},
        )
        raise
    finally:
        try:
            cleanup(state_path, state)
        finally:
            signal.signal(signal.SIGINT, previous_int)
            signal.signal(signal.SIGTERM, previous_term)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Launch, execute and always clean up an EC2 latency hunt"
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
    parser.add_argument("--instance-types", required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--binary-sha256", required=True)
    parser.add_argument("--remote-script", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--state-file", type=Path, required=True)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--smoke-duration-seconds", type=int, default=60)
    return parser.parse_args()


if __name__ == "__main__":
    execute(parse_args())
