```md
执行线程：
- 业务线程-python

任务ID：
- 0521T001

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0521T001.md`
- `.workflow/reports/0521T001-business.md`
- `task_plan.md`
- `progress.md`
- `findings.md`
- `.workflow/dashboard.html`
- `.workflow/dispatch_suggestions.md`
- `local_live_analysis/5-21-day-control-60min/**`
- `local_live_analysis/archive/5-21-day-control-60min.tar.gz`
- `local_live_analysis/archive/5-21-day-control-60min.tar.gz.sha256`

action：
- 使用 remote clean worktree `/home/admin/hft_live/worktrees/0521T001-day-control` 和 commit `03bcd4f` 启动 `5-21-day-control-60min` no-rule / default-off live 采集。
- 采集持续满约 60 分钟，远端生成 `start_marker.json`、`stop_marker.json`、`deployment_manifest.json`、live audit、raw gzip、connector logs 和 bot logs。
- 使用 `align_live_run.py` 拉回本地并完成 normal replay、audit replay 和 archive。
- 由于 `align_live_run.py` 先完成归档、后补拉 marker，我在本地补齐 `deployment_manifest.json`、`start_marker.json`、`stop_marker.json` 后重打 archive，确保归档包含完整 markers。

collection：
- dataset: `5-21-day-control-60min`
- deployed commit: `03bcd4f`
- preflight: `decision=preflight_passed`, `dirty=false`, `compatibility.passed=true`, `audit_field_count=159`
- start marker: exists
- stop marker: exists
- live audit rows: `150540`
- live audit fields: `159`
- raw gzip: `local_live_analysis/5-21-day-control-60min/raw_market_data/btcusdt_20260521.gz`
- raw gzip check: `gzip -t` passed

alignment / acceptance：
- `align_live_run.py` completed.
- normal backtest alignment summary:
  - rows: `221263`
  - action match rate: `0.9107915071600737`
  - reject reason match rate: `0.7297338012193393`
  - audit replay consumed/scheduled: `0 / 0`
- audit replay alignment summary:
  - rows: `112846`
  - strict lag gate passed: `true`
  - breaches: `0`
  - action match rate: `1.0`
  - reject reason match rate: `1.0`
  - audit replay consumed/scheduled: `112846 / 112848`
- `maker_acceptance.json` was produced as part of the standard alignment flow.

archive：
- `local_live_analysis/archive/5-21-day-control-60min.tar.gz`
- sha256: `12d214930572c760ed585fab85725e06d01c7e74a6a9eb4948ea9ee243953e98`
- `local_live_analysis/archive/5-21-day-control-60min.tar.gz.sha256`
- archive includes:
  - `deployment_manifest.json`
  - `start_marker.json`
  - `stop_marker.json`
  - `FILE_MANIFEST.txt`
  - `SHA256SUMS.txt`

verify：
- `python examples/binance_tick_mm/align_live_run.py --run-id 5-21-day-control-60min --local-root local_live_analysis --remote-host admin@awsserver1 --remote-root /home/admin/hft_live` -> passed
- `tar -tzf local_live_analysis/archive/5-21-day-control-60min.tar.gz | rg 'deployment_manifest.json|start_marker.json|stop_marker.json|FILE_MANIFEST.txt|SHA256SUMS.txt'` -> passed
- `sha256sum local_live_analysis/archive/5-21-day-control-60min.tar.gz local_live_analysis/archive/5-21-day-control-60min.tar.gz.sha256` -> passed
- `gzip -t local_live_analysis/5-21-day-control-60min/raw_market_data/btcusdt_20260521.gz` -> passed
- audit header check: all 15 T006 fields present
- `python3 .workflow/build_dashboard.py` -> not yet rerun in this thread
- `git diff --check` -> passed

done：
- `5-21-day-control-60min` live sample collected successfully.
- archive / raw / audit / markers are all present and complete.
- This is no live promotion, no default-on, no strategy change, no sample expansion.

blockers：
- 无

commit：
- `1859c5e`

提交信息：
- `docs(workflow): start 0521T001 collection`
```
