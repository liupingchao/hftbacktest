# 线程回报

执行线程：
- 业务线程-live-evidence

任务ID：
- 0717T002

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0717T002.md`
- `.workflow/reports/0717T002-business.md`
- `docs/qa-acceptance-report.md`
- `task_plan.md`
- `progress.md`
- `findings.md`
- local artifact root: `local_live_analysis/cross_exchange_controlled_role_evidence_0717T002_20260717T045820Z/`

action：
- Confirmed SSM connectivity for EC2 instance `i-02c64c088f311cbc1`.
- Confirmed remote preflight as `admin`: env file exists, Hyperliquid SDK imports, watcher/orchestrator compile and help paths pass, and no pre-existing live process was active.
- Synced the 0717T001 orchestrator to awsserver1 because remote repo was still at `e97053960d052cb0155333e0bc85fbe48f2e095b` and did not contain the new file.
- Launched detached SSM-first live orchestrator at `2026-07-17T04:58:41Z`.
- Ran 3 sequential windows under the conservative envelope:
  - Hyperliquid `BTC`
  - post-only `Alo`
  - max order size `0.005 BTC`
  - max submissions `2` per window
  - quote hold `3s`
  - wait `10s`
  - fast `l2Book`
- Pulled the complete remote artifact root back locally.
- Added a final root-level read-only `open_orders()` proof after completion.

remote：
- run root: `/home/admin/hftbacktest-cross-exchange-artifacts/cross_exchange_controlled_role_evidence_0717T002_20260717T045820Z`
- local root: `local_live_analysis/cross_exchange_controlled_role_evidence_0717T002_20260717T045820Z/`
- launch SSM command id: `dcf16b3a-8da9-4238-b7be-75c3d853b3fa`
- final root open-orders proof command id: `540146d6-b9e3-4d05-9fc2-9db57439a91d`

result：
- Orchestrator state: `complete`.
- Window 01:
  - started `2026-07-17T04:58:41Z`
  - ended `2026-07-17T05:11:54Z`
  - runner return code `0`
  - independent open orders `0`
  - order intents `1`
  - resting lifecycle rows `1`
  - fill ledger rows `0`
  - liquidity-role rows `0`
- Window 02:
  - started `2026-07-17T05:11:54Z`
  - ended `2026-07-17T05:36:06Z`
  - runner return code `0`
  - independent open orders `0`
  - order intents `1`
  - resting lifecycle rows `1`
  - fill ledger rows `0`
  - liquidity-role rows `0`
- Window 03:
  - started `2026-07-17T05:36:06Z`
  - ended `2026-07-17T05:50:12Z`
  - runner return code `0`
  - independent open orders `0`
  - order intents `2`
  - resting lifecycle rows `0`
  - fill ledger rows `0`
  - liquidity-role rows `0`
  - both attempts were post-only immediate-match rejects.
- Final root open orders:
  - proof at `2026-07-17T06:00:15Z`
  - final open orders count `0`
  - final open orders empty `true`
- Total real order intents in artifacts: `4`.
- Total live fill ledger rows: `0`.
- Total fill liquidity role rows: `0`.

verify：
- SSM online and EC2 instance checks passed before launch.
- Remote `py_compile` and `--help` checks passed before launch.
- Remote Hyperliquid SDK import passed before launch.
- Local artifact parse:
  - JSON parsed: `109`
  - JSON errors: `0`
  - CSV parsed: `117`
  - CSV errors: `0`
- Remote sha manifest verification after pullback:
  - manifest lines: `241`
  - sha ok: `241`
  - sha missing: `0`
  - sha bad: `0`
- Final root `open_orders()` proof:
  - `final_open_orders_count=0`
  - `final_open_orders_empty=true`

done：
- The SSM-first live collection contract worked: no long-lived SSH session was needed for execution truth, and all three windows completed with recoverable status/heartbeat/window artifacts.
- The run did not produce any fills, so it did not produce maker/taker role evidence.
- P0 fill source / liquidity-role evidence remains blocked on absence of fills.

blockers：
- No fill role evidence was produced because there were zero live fills.
- Fee/PnL calibration, maker fill count, fill-rate calibration, maker viability, T012, promotion, and final MVP pass remain unsupported.

deviations：
- Remote repo commit remained `e97053960d052cb0155333e0bc85fbe48f2e095b`; only the new 0717T001 orchestrator/test files were copied to remote to make SSM-first execution possible.
- The current CLI path does not expose a `--max-loss 1` or `--max-position-delta 0.01` flag; those remained task-envelope / post-run validation constraints, not newly implemented runtime CLI controls.
- Raw artifacts are ignored by Git via `local_live_analysis*/`; only reports and workflow docs are committed.

commit：
- TBD

提交信息：
- TBD
