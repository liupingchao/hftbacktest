# 线程回报

执行线程：
- 业务线程-python/live-awsserver1

任务ID：
- 0715T001

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0715T001.md`
- `.workflow/reports/0715T001-business.md`
- `local_live_analysis/cross_exchange_interval_coverage_repaired_live_evidence_0715T001_20260715T132113Z/`
- `task_plan.md`
- `progress.md`
- `findings.md`

action：
- 使用既有正式任务文件 `.workflow/tasks/0715T001.md`，未创建重复任务。
- 按 UTC gate 执行 preflight：
  - target UTC: `2026-07-15T13:15:00Z`
  - heartbeat trigger: `2026-07-15T13:15:07.832Z`
  - 本地 preflight time: `2026-07-15T13:17:21Z`
  - awsserver1 sync/preflight complete: `2026-07-15T13:20:06Z`
- 验证 `0714T005` QA 为 `已通过`。
- 验证 local/origin/amdserver 对齐到 `26dae0ada9038df4ef17b9379230260d4b3ca3fe`。
- 发现 `awsserver1` 仍指向旧 bundle origin，缺少 `.workflow/tasks/0715T001.md`；按任务规则由本地创建 bundle 并同步 `awsserver1:/home/admin/hftbacktest-cross-exchange` 到 `26dae0ada9038df4ef17b9379230260d4b3ca3fe`。
- 在 `awsserver1` 顺序执行三段 controlled live window，未并行，未额外 retry。
- 每段均使用：
  - `--event-driven-edge-gate-live`
  - `--hyperliquid-l2book-fast`
  - `--watcher-seconds 1800`
  - `--max-order-size 0.005`
  - `--max-real-order-submissions 2`
  - `--quote-hold-seconds 3`
  - `--wait-seconds 10`
  - `--artifact-task-id 0715T001`
- 每段完成后执行独立 read-only open-orders check，并写入 `independent_remote_open_orders_check.json`。
- Pull back 远端 artifact 到本地并生成 local validation、sha256 reconciliation、boundary manifest、window lifecycle summary 和 validation report。

remote artifact：
- `/home/admin/hftbacktest-cross-exchange-artifacts/cross_exchange_interval_coverage_repaired_live_evidence_0715T001_20260715T132113Z/`

local artifact：
- `local_live_analysis/cross_exchange_interval_coverage_repaired_live_evidence_0715T001_20260715T132113Z/`

window results：
- `window_01`
  - actual UTC: `2026-07-15T13:21:13Z` to `2026-07-15T13:28:04Z`
  - lifecycle: `submitted_resting_no_fill`
  - live submissions: `1`
  - order status: `resting`
  - fill count: `0`
  - maker fill count: `0`
  - final open orders: `0`
  - independent final open orders empty: `true`
  - interval public trade rows: `6`
  - public stream coverage rows: `1`
- `window_02`
  - actual UTC: `2026-07-15T13:30:20Z` to `2026-07-15T13:31:19Z`
  - lifecycle: `submitted_resting_no_fill`
  - live submissions: `2`
  - order status: `error,resting`
  - fill count: `0`
  - maker fill count: `0`
  - final open orders: `0`
  - independent final open orders empty: `true`
  - interval public trade rows: `18`
  - public stream coverage rows: `1`
- `window_03`
  - actual UTC: `2026-07-15T13:32:17Z` to `2026-07-15T13:35:38Z`
  - lifecycle: `submitted_no_resting_reject_or_error_no_fill`
  - live submissions: `2`
  - order status: `error,error`
  - fill count: `0`
  - maker fill count: `0`
  - final open orders: `0`
  - independent final open orders empty: `true`
  - interval public trade rows: `0`
  - public stream coverage rows: `0`

verify：
- Local preflight:
  - `git status --short --branch`
  - `git rev-parse HEAD origin/cross-exchange`
  - `rg -n "0714T005|状态：|已通过" .workflow/reports/0714T005-qa.md docs/qa-acceptance-report.md`
- amdserver preflight:
  - `ssh amdserver 'cd ~/project/hftbacktest && git status --short --branch && git rev-parse HEAD origin/cross-exchange && test -f .workflow/tasks/0715T001.md'`
- awsserver1 sync/preflight:
  - local git bundle sync to `/home/admin/hftbacktest-cross-exchange-0715T001-26dae0a-v2.bundle`
  - `git reset --hard origin/cross-exchange`
  - `git clean -fd`
  - `python -m py_compile examples/hyperliquid/hyperliquid_tiny_live_m2_public_watcher.py`
  - `python examples/hyperliquid/hyperliquid_tiny_live_m2_public_watcher.py --help`
  - `git diff --check`
- Post-run validation:
  - JSON/CSV parse errors: `0`
  - local/remote sha256 reconciliation: `pass`
  - `remote_sha256_manifest.txt` excluded from strict matching because it is self-referential.
  - final open orders all empty: `true`
  - boundary validation status: `pass`

done：
- `0715T001` controlled live evidence ran exactly three sequential windows under the authorized conservative envelope.
- Total live submissions across windows: `5`.
- Total fills: `0`.
- Total maker fills: `0`.
- All final open-orders checks are empty.
- Repaired interval evidence is useful for windows 1 and 2:
  - both include resting lifecycle and public-stream coverage rows.
  - window 3 never reached resting, so interval evidence is correctly empty / not applicable.
- This task does not support fill probability, queue priority, fee/rebate, realized PnL, profitability, stable PnL, maker viability, `T012`, promotion, final MVP pass, or parameter expansion.

blockers：
- No execution blocker remains for this task.
- Analytical blocker remains: no fill occurred, so fee/PnL calibration and realized-PnL proof remain unsupported.

commit：
- 待提交

提交信息：
- 待提交
