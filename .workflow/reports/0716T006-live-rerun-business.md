# 线程回报

执行线程：
- 业务线程-live-awsserver1

任务ID：
- 0716T006

状态：
- 阻塞

是否进行QA验收：
- 是

QA说明：
- 本报告覆盖 0716T006 在补齐 controller live authorization 后的 live rerun、awsserver1 SSH/SSM recovery、完整 artifact pullback 和三窗口验证。
- Connectivity blocker 已恢复；当前阻塞原因迁移为：三窗口均无 fill，因此没有可验收的 fill source / maker-taker role evidence。

files：
- `.workflow/tasks/0716T006.md`
- `.workflow/reports/0716T006-live-rerun-business.md`
- `local_live_analysis/cross_exchange_controlled_role_evidence_0716T006/`
- `local_live_analysis/cross_exchange_controlled_role_evidence_0716T006_20260716T073133Z_full/`
- `task_plan.md`
- `progress.md`
- `findings.md`

authorization：
- Host: `awsserver1`
- Remote repo: `/home/admin/hftbacktest-cross-exchange`
- Interpreter: `/home/admin/.venvs/hyperliquid-sdk-0618T002/bin/python`
- Env file boundary: `/home/admin/XEMM_rust_latest/.env`, without printing/copying/pulling secrets
- Authorized code source: `cross-exchange / a5431d8b24da7d77671148d316f789b0b25cf3f8`
- Live rerun authorization commit: `e97053960d052cb0155333e0bc85fbe48f2e095b`
- Venue/symbol: Hyperliquid `BTC`
- Post-only behavior: `Alo`
- Windows: `3` sequential windows, each bounded by `1800s`
- Max order size: `0.005 BTC`
- Max submissions: `2` per window
- Max position delta: `0.01 BTC`
- Max loss: `1 USDC`
- Real order submit/cancel: authorized under this envelope only

action：
- Reopened 0716T006 from the initial `blocked_missing_live_authorization` gate after controller supplied the live envelope.
- Committed formal authorization record:
  - `e970539 / Authorize 0716T006 live rerun envelope`
- Synced `awsserver1:/home/admin/hftbacktest-cross-exchange` to the authorized local source.
- Remote preflight passed:
  - `git diff --check`
  - watcher `py_compile`
  - watcher `--help`
  - Hyperliquid SDK import
- Executed three authorized live windows.
- After Window 3, awsserver1 connectivity failed. Recovery actions completed after controller reboot/authorization:
  - EC2 Instance Connect `SendSSHPublicKey` succeeded.
  - SSH recovered for Debian user `admin`.
  - EC2 SSM instance profile `CodexEC2SSMCoreProfile` with role `CodexEC2SSMCoreRole` is associated.
  - `amazon-ssm-agent` was installed/enabled and is active.
  - `aws ssm get-connection-status` reports `connected`.
  - Remote process check found no remaining `0716T006` / `hyperliquid_tiny_live` strategy process.
  - Read-only Hyperliquid `open_orders()` recovery proof returned `0`.
- Pulled the complete remote artifact package:
  - remote root: `/home/admin/hftbacktest-cross-exchange-artifacts/cross_exchange_controlled_role_evidence_0716T006_20260716T073133Z`
  - local root: `local_live_analysis/cross_exchange_controlled_role_evidence_0716T006_20260716T073133Z_full/`
  - recursive file count: `233`
  - `remote_sha256_manifest.txt` rows: `233`

window results：
- Window 1:
  - started at `2026-07-16T07:31:33Z`
  - order statuses: `error,resting`
  - order intent rows: `2`
  - live fill ledger rows: `0`
  - fill liquidity role evidence rows: `0`
  - real order endpoint called: `true`
  - real cancel endpoint called: `true`
  - shutdown proof: `pass`
  - final open orders: `0`
- Window 2:
  - completed before recovery with runner rc `0`
  - order statuses: `error,resting`
  - order intent rows: `2`
  - live fill ledger rows: `0`
  - fill liquidity role evidence rows: `0`
  - real order endpoint called: `true`
  - real cancel endpoint called: `true`
  - independent open-orders proof: `0`
  - shutdown proof: `pass`
  - final open orders: `0`
- Window 3:
  - run completed at `2026-07-16T08:17:07Z`
  - order statuses: `error,resting`
  - order intent rows: `2`
  - live fill ledger rows: `0`
  - fill liquidity role evidence rows: `0`
  - real order endpoint called: `true`
  - real cancel endpoint called: `true`
  - independent open-orders proof: `0`
  - shutdown proof: `pass`
  - final open orders: `0`

verify：
- `ssh awsserver1 'date -u ...; systemctl is-active ssh; systemctl is-active amazon-ssm-agent; pgrep -af "0716T006|hyperliquid_tiny_live"'`
  - SSH active
  - SSM agent active
  - no strategy process observed
- `aws ssm get-connection-status --profile codex-cli --region ap-northeast-1 --target i-02c64c088f311cbc1`
  - status `connected`
- Full artifact parse:
  - JSON files parsed: `101`, errors `0`
  - CSV files parsed: `117`, errors `0`
- Window semantic validation:
  - all three windows have `real_order_endpoint_called=true`
  - all three windows have `real_cancel_endpoint_called=true`
  - all three windows have `shutdown_proof_status=pass`
  - all three windows have `final_open_orders_count=0`
  - all three windows have `fill_count=0`, `ledger_fill_rows=0`, and `role_rows=0`
- `live_rerun_recovery_final_summary.json` parses.

done：
- Controlled live evidence package was recovered and validated for all three windows.
- Connectivity is no longer the active blocker.
- Role/source-path evidence status:
  - not accepted; there were no fills, so `fill_liquidity_role_evidence.csv` and `live_fill_ledger.csv` contain no role/fill rows to prove maker/taker source-path handling.
- Final route:
  - `route_to_controlled_evidence_rerun_or_explicit_downgrade_no_fill_role_evidence`
- Unsupported claims:
  - no accepted fill source / maker-taker role evidence
  - no fee/PnL calibration
  - no maker viability
  - no T004 public shadow unlock unless controller explicitly downgrades the role-evidence requirement
  - no T012, promotion, or final MVP pass

blockers：
- `no_fill_role_evidence_absent`
- Three windows completed, but zero fills means there is no maker/taker role attribution row to accept.

required next decision：
- Either run a separately authorized controlled evidence rerun designed to obtain at least one role-attributable fill under bounded risk, or explicitly downgrade the first-three sequence so T004 public shadow can proceed without live fill role evidence.

commit：
- TBD

提交信息：
- TBD
