# Cross-Exchange Live Collection Resilience

## Purpose

Future live evidence runs should not depend on a long-lived public SSH session.
The live strategy process must be recoverable through remote status files and
SSM even if local SSH/scp disconnects.

## Current Fix

`examples/hyperliquid/cross_exchange_live_remote_orchestrator.py` wraps the
existing Hyperliquid live watcher without changing strategy behavior.

It provides:

- a nonblocking live lock: `/tmp/hftbacktest_live_test.lock`
- `run_status.json`
- `heartbeat.json`
- `orchestrator_events.jsonl`
- per-window `window_status.json`
- per-window `runner_stdout.log` and `runner_stderr.log`
- per-window read-only `independent_remote_open_orders_check.json`
- `abort_manifest.json` on fail/abort
- `run_complete.json` on success
- `remote_sha256_manifest.txt`

## SSM-First Launch Pattern

Use SSM RunCommand as the primary control plane. SSH can still be used for
manual inspection when available, but live execution should not rely on SSH
remaining connected.

Example command shape:

```bash
aws ssm send-command \
  --profile codex-cli \
  --region ap-northeast-1 \
  --instance-ids i-02c64c088f311cbc1 \
  --document-name AWS-RunShellScript \
  --comment '0717 live evidence orchestrator' \
  --parameters commands='[
    "cd /home/admin/hftbacktest-cross-exchange",
    "/home/admin/.venvs/hyperliquid-sdk-0618T002/bin/python examples/hyperliquid/cross_exchange_live_remote_orchestrator.py --task-id <TASK_ID> --remote-repo /home/admin/hftbacktest-cross-exchange --env-file /home/admin/XEMM_rust_latest/.env --output-root /home/admin/hftbacktest-cross-exchange-artifacts --windows 3 --window-seconds 1800 --max-order-size 0.005 --max-submissions 2 --quote-hold-seconds 3 --wait-seconds 10 --hyperliquid-l2book-fast"
  ]'
```

For a separately detached launch, wrap the command with `nohup` or
`systemd-run`. The orchestrator itself remains the evidence source of truth.

## Recovery Order

If SSH disconnects:

1. Check SSM connection status.
2. Read `run_status.json` and `heartbeat.json` through SSM RunCommand.
3. Check for any remaining orchestrator/watcher process.
4. Read `independent_remote_open_orders_check.json` for each completed window.
5. Pull the artifact root when SSH/scp is available, or use SSM/S3 in a future task.
6. Validate JSON/CSV and `remote_sha256_manifest.txt` before QA.

## Deferred Work

This task does not add:

- S3 artifact upload/download.
- a permanent systemd service or timer.
- AWS security group, VPC, subnet, or IAM hardening.
- strategy parameter changes.
- quote policy changes.
- order size, max submission, max loss, or threshold changes.
- T004 public shadow unlock.
- fee/PnL calibration.
