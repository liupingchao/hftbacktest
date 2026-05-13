```md
执行线程：
- 测试线程

任务ID：
- 0513T008

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0513T008.md`
- `.workflow/reports/0513T008-business.md`
- `task_plan.md`
- `progress.md`
- `findings.md`
- `.workflow/dashboard.html`
- `.workflow/dispatch_suggestions.md`
- `local_live_analysis/5-13-day-control-30min/**`
- `local_live_analysis/archive/5-13-day-control-30min.tar.gz`
- `local_live_analysis/archive/5-13-day-control-30min.tar.gz.sha256`

action：
- 创建并执行 `0513T008`，目标为采集 fresh no-rule control 样本 `5-13-day-control-30min`。
- 本地提交并推送部署 commit `f228950`。
- 远端使用 clean git worktree `/home/admin/hft_live/worktrees/0513T008-control` 指向 `f228950`，避免污染远端主工作树。
- 远端 run dir：`/home/admin/hft_live/runs/5-13-day-control-30min/`。
- 使用 T004 `run_live.sh` 标准流程启动，显式设置 `PYTHON_BIN=/home/admin/hft_live/venv/bin/python`。
- 采集完成后拉回 artifacts，运行 `align_live_run.py`、`maker_acceptance.py`、T007 full-run sidecar 和 as-of decision join。
- 写入 `local_live_analysis/5-13-day-control-30min/t008_sample_classification.md`。
- 刷新 archive 和 sha256。

verify：
- Local pre-deploy:
  - `python -m pytest examples/binance_tick_mm/test_deploy_preflight.py` -> `5 passed`
  - `python -m pytest examples/binance_tick_mm/test_binance_top5_provenance.py` -> `4 passed`
  - `python examples/binance_tick_mm/deploy/preflight_live_run.py --help` -> 通过
  - `python examples/binance_tick_mm/binance_top5_provenance.py --help` -> 通过
  - `bash -n examples/binance_tick_mm/deploy/run_live.sh` -> 通过
- Remote/live:
  - T004 preflight passed: `commit=f228950`, `schema_fields=144`
  - `deployment_manifest.json`: `decision=preflight_passed`, `git.dirty=false`, `compatibility.passed=true`
  - `start_marker.json`: exists
  - `stop_marker.json`: exists, `exit_code=0`
  - start marker UTC: `2026-05-13T08:30:12Z`
  - stop marker UTC: `2026-05-13T09:00:13Z`
  - stopped local: `2026-05-13T18:00:25+0900`
  - remote audit rows: `63321` including header
  - remote raw gzip: `19M`, `gzip -t` passed
- `align_live_run.py`:
  - command: `python examples/binance_tick_mm/align_live_run.py --run-id 5-13-day-control-30min --local-root local_live_analysis --remote-host admin@awsserver1 --remote-root /home/admin/hft_live`
  - exit 0
  - live rows: `63320`
  - decision rows: `47499`
  - audit replay consumed/scheduled: `47496 / 47499`
  - archive: `local_live_analysis/archive/5-13-day-control-30min.tar.gz`
- `maker_acceptance.py`:
  - `passed=true`
  - hard failures: `[]`
  - common rows: `47496`
  - action/planned/reject/throttle match rate: `1.0 / 1.0 / 1.0 / 1.0`
  - working-order semantic/blocking mismatch: `0 / 0`
  - API throttle mismatch / target tick mismatch: `0 / 0`
  - strict replay lag breach/drop/fail: `0 / 0 / 0`
- T007 full-run sidecar:
  - command: `python examples/binance_tick_mm/binance_top5_provenance.py build-sidecars --input-gz local_live_analysis/5-13-day-control-30min/raw_market_data/btcusdt_20260513.gz --out-dir local_live_analysis/5-13-day-control-30min/t007_full_run_sidecar --sample-id 5-13-day-control-30min --symbol BTCUSDT --tick-size 0.1 --buffer-size 8000000`
  - generated `data.npz`, `raw_provenance.csv`, `raw_to_npz_mapping.csv`, `top5_sidecar.csv`, `sidecar_manifest.json`, `metrics.json`
  - raw message count: `461402`
  - npz row count: `2418372`
  - raw messages with npz rows: `105579`
  - raw message mapping coverage: `0.2288221551`，主要因为 bookTicker 默认不写入 standard npz
  - final data row mapping coverage after correction: `1.0`
  - depth `pu` mismatch count: `0`
  - snapshot alignment status: `present`
  - first valid update aligned: `false`
  - top5 row count: `67321`
  - bookTicker/depth BBO match/mismatch: `67306 / 11`
- T007 decision join:
  - command: `python examples/binance_tick_mm/binance_top5_provenance.py join-decisions --audit-csv local_live_analysis/5-13-day-control-30min/audit_live_5-13-day-control-30min.csv --top5-csv local_live_analysis/5-13-day-control-30min/t007_full_run_sidecar/top5_sidecar.csv --out-csv local_live_analysis/5-13-day-control-30min/t007_full_run_sidecar/joined_decisions.csv --max-age-ms 250`
  - decision count: `47499`
  - joined decision count: `47499`
  - decision join coverage: `1.0`
  - future join count: `0`
  - missing join count: `0`
  - stale join count: `432`
  - gap-crossed join count: `47499`
  - top5 join age p50/p90/p99 ms: `13.602 / 24.206 / 28.134`
- Archive:
  - `local_live_analysis/archive/5-13-day-control-30min.tar.gz`
  - sha256: `d7291afe0cb547663d4aa8e4cc9a175bfd06a3e7fcffea9d9eab82cc70b5c033`

done：
- `5-13-day-control-30min` 已完成采集、拉回、align、maker acceptance、T007 full-run sidecar/join、archive。
- 最终分类：`pricing_research_candidate`。
- 分类边界：只适合 compressed action-path acceptance 和 limited BBO/bookTicker/compressed-mid pricing sanity。
- 不适合升级为 `queue_fill_proxy_candidate`。
- 不适合直接用于 top5 microprice / top5 OFI proxy 研究，因为 `first_valid_update_aligned=false` 且 joined decisions 全部 `gap_crossed=47499/47499`。
- 本任务不是策略收益证明，不授权 live promotion，不授权策略规则变更。

blockers：
- 无阻塞；但数据质量上存在后续必须解决的问题：T007 reconstructed top5 book 未通过 Binance first-valid snapshot/update alignment，导致所有 decision join 标记为 gap-crossed。

commit：
- f228950

提交信息：
- test(binance): start 30min control collection task
```
