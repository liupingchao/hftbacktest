#!/usr/bin/env python3
import argparse
import gzip
import hashlib
import json
import os
import shlex
import subprocess
import tarfile
import time
from pathlib import Path

import botocore.session
from botocore.config import Config


def run(*command, check=True):
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if check and result.returncode:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(command)}\n"
            f"{result.stderr.strip()}"
        )
    return result


def error_has_any(result, markers):
    message = f"{result.stdout}\n{result.stderr}".lower()
    return any(marker.lower() in message for marker in markers)


def bucket_exists(region, bucket):
    result = run(
        "aws",
        "s3api",
        "head-bucket",
        "--region",
        region,
        "--bucket",
        bucket,
        check=False,
    )
    if result.returncode == 0:
        return True
    if error_has_any(
        result, ("404", "Not Found", "NoSuchBucket", "NotFound")
    ):
        return False
    raise RuntimeError(
        f"unable to determine whether build bucket {bucket} exists: "
        f"{result.stderr.strip()}"
    )


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_source_archive(repo_root, linux_lock, output):
    files = [
        (repo_root / "latency-probe" / "Cargo.toml", "Cargo.toml"),
        (linux_lock, "Cargo.lock"),
        (repo_root / "latency-probe" / "src" / "lib.rs", "src/lib.rs"),
        (repo_root / "latency-probe" / "src" / "main.rs", "src/main.rs"),
    ]

    def normalize(info):
        info.mtime = 0
        info.uid = 0
        info.gid = 0
        info.uname = ""
        info.gname = ""
        info.mode = 0o644
        return info

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as zipped:
            with tarfile.open(
                fileobj=zipped, mode="w", format=tarfile.PAX_FORMAT
            ) as bundle:
                for source, arcname in sorted(files, key=lambda item: item[1]):
                    bundle.add(
                        source,
                        arcname=arcname,
                        recursive=False,
                        filter=normalize,
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
        ExpiresIn=7200,
    )


def wait_command(region, command_id, instance_id, timeout_seconds=1800):
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        result = run(
            "aws",
            "ssm",
            "get-command-invocation",
            "--region",
            region,
            "--command-id",
            command_id,
            "--instance-id",
            instance_id,
            "--output",
            "json",
        )
        row = json.loads(result.stdout)
        if row["Status"] == "Success":
            return row
        if row["Status"] in {"Cancelled", "Failed", "TimedOut", "Cancelling"}:
            raise RuntimeError(
                f"builder command {command_id} failed: {row['Status']}\n"
                f"{row.get('StandardErrorContent', '')}"
            )
        time.sleep(10)
    raise RuntimeError(f"builder command timeout: {command_id}")


def main():
    parser = argparse.ArgumentParser(
        description="Build the Linux x86_64 hunt probe on an existing SSM host"
    )
    parser.add_argument("--region", default="ap-northeast-1")
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--linux-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--source-archive-output", type=Path, required=True)
    args = parser.parse_args()

    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    bucket = f"hft-latency-build-{run_id.lower()}-{os.getpid()}"
    source_key = "source/latency-probe-source.tar.gz"
    binary_key = "artifact/latency-probe-linux-x86_64"
    command_id = None
    source_archive = args.source_archive_output
    create_source_archive(args.repo_root, args.linux_lock, source_archive)
    source_sha = sha256(source_archive)
    try:
        source_archive_path = str(source_archive.relative_to(args.repo_root))
    except ValueError:
        source_archive_path = str(source_archive)
    try:
        run(
            "aws",
            "s3api",
            "create-bucket",
            "--region",
            args.region,
            "--bucket",
            bucket,
            "--create-bucket-configuration",
            f"LocationConstraint={args.region}",
        )
        run(
            "aws",
            "s3",
            "cp",
            "--region",
            args.region,
            str(source_archive),
            f"s3://{bucket}/{source_key}",
        )
        source_url = presigned_url(
            args.region, bucket, source_key, "get_object"
        )
        artifact_url = presigned_url(
            args.region, bucket, binary_key, "put_object"
        )
        remote_root = f"/tmp/ec2-latency-build-{run_id}"
        command = " && ".join(
            [
                "export DEBIAN_FRONTEND=noninteractive",
                "apt-get update",
                "apt-get install -y build-essential ca-certificates curl libssl-dev pkg-config",
                f"rm -rf {shlex.quote(remote_root)}",
                f"mkdir -p {shlex.quote(remote_root)}",
                f"curl -fsS {shlex.quote(source_url)} -o {shlex.quote(remote_root + '/source.tar.gz')}",
                f"tar -C {shlex.quote(remote_root)} -xzf {shlex.quote(remote_root + '/source.tar.gz')}",
                "curl -fsSL https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain 1.93.0",
                f"cd {shlex.quote(remote_root)}",
                "/root/.cargo/bin/cargo +1.93.0 build --release --locked",
                "sha256sum target/release/latency-probe",
                f"curl -fsS -X PUT --upload-file target/release/latency-probe {shlex.quote(artifact_url)}",
                f"rm -rf {shlex.quote(remote_root)}",
            ]
        )
        result = run(
            "aws",
            "ssm",
            "send-command",
            "--region",
            args.region,
            "--instance-ids",
            args.instance_id,
            "--document-name",
            "AWS-RunShellScript",
            "--parameters",
            json.dumps(
                {"commands": [command], "executionTimeout": ["1800"]}
            ),
            "--timeout-seconds",
            "1800",
            "--query",
            "Command.CommandId",
            "--output",
            "text",
        )
        command_id = result.stdout.strip()
        invocation = wait_command(
            args.region, command_id, args.instance_id
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        run(
            "aws",
            "s3",
            "cp",
            "--region",
            args.region,
            f"s3://{bucket}/{binary_key}",
            str(args.output),
        )
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        head = run(
            "aws",
            "s3api",
            "head-object",
            "--region",
            args.region,
            "--bucket",
            bucket,
            "--key",
            binary_key,
            "--output",
            "json",
        )
        content_length = json.loads(head.stdout)["ContentLength"]
        if content_length != args.output.stat().st_size:
            raise RuntimeError(
                "builder artifact size mismatch: "
                f"s3={content_length} local={args.output.stat().st_size}"
            )
        args.receipt.write_text(
            json.dumps(
                {
                    "schema_version": "ec2-latency-hunt-build-v2",
                    "builder_instance_id": args.instance_id,
                    "command_id": command_id,
                    "source_archive_path": source_archive_path,
                    "source_archive_sha256": source_sha,
                    "cargo_toml_sha256": sha256(
                        args.repo_root / "latency-probe" / "Cargo.toml"
                    ),
                    "cargo_lock_sha256": sha256(args.linux_lock),
                    "lib_rs_sha256": sha256(
                        args.repo_root / "latency-probe" / "src" / "lib.rs"
                    ),
                    "main_rs_sha256": sha256(
                        args.repo_root / "latency-probe" / "src" / "main.rs"
                    ),
                    "binary_sha256": sha256(args.output),
                    "binary_bytes": args.output.stat().st_size,
                    "rust_toolchain": "1.93.0",
                    "status": invocation["Status"],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    finally:
        errors = []
        result = run(
            "aws",
            "s3",
            "rm",
            "--region",
            args.region,
            f"s3://{bucket}",
            "--recursive",
            check=False,
        )
        if result.returncode:
            errors.append(result.stderr.strip())
        result = run(
            "aws",
            "s3api",
            "delete-bucket",
            "--region",
            args.region,
            "--bucket",
            bucket,
            check=False,
        )
        if result.returncode:
            errors.append(result.stderr.strip())
        try:
            if bucket_exists(args.region, bucket):
                errors.append(f"build bucket still exists: {bucket}")
        except BaseException as error:
            errors.append(str(error))
        if errors:
            raise RuntimeError(f"build bucket cleanup failed: {errors}")


if __name__ == "__main__":
    main()
