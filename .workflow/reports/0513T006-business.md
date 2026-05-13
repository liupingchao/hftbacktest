```md
执行线程：
- 业务线程-python

任务ID：
- 0513T006

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0513T006.md`
- `.workflow/reports/0513T006-business.md`
- `task_plan.md`
- `progress.md`
- `findings.md`
- `.workflow/dashboard.html`
- `.workflow/dispatch_suggestions.md`
- `local_live_analysis/step2_market_data_baseline_0513T006/sample_inventory.json`
- `local_live_analysis/step2_market_data_baseline_0513T006/latency_summary.json`
- `local_live_analysis/step2_market_data_baseline_0513T006/latency_summary.csv`
- `local_live_analysis/step2_market_data_baseline_0513T006/market_data_integrity.json`
- `local_live_analysis/step2_market_data_baseline_0513T006/top5_mismatch_buckets.csv`
- `local_live_analysis/step2_market_data_baseline_0513T006/source_provenance_summary.json`
- `local_live_analysis/step2_market_data_baseline_0513T006/sample_usability_matrix.csv`
- `local_live_analysis/step2_market_data_baseline_0513T006/step2_baseline_report.md`

action：
- 按 `0513T005` 计划完成 Step 2 read-only latency / market-data integrity baseline。
- 只读取现有本地样本和已有分析产物，生成 latency、market-data integrity、provenance、sample usability artifacts。
- 覆盖样本：
  - `5-13-day-control-15min`
  - `5-11-night-active`
  - `5-10-day-control-1h-06`
  - `5-9-noon`
  - `5-9-small`
- 未启动 live，未采集新样本，未运行 Stage 6J candidate replay/sweep，未修改 strategy/deploy/core/converter/connector 源码。

verify：
- `find local_live_analysis/step2_market_data_baseline_0513T006 -maxdepth 1 -type f -printf '%f\n' | sort`
  - 已确认 8 个 required artifacts 均存在。
- `python3 .workflow/build_dashboard.py`
  - 已通过，刷新 dashboard 与 dispatch suggestions。
- 人工边界检查：
  - 无 live。
  - 无 replay/sweep。
  - 本任务没有修改 `examples/binance_tick_mm/*.py`、`examples/binance_tick_mm/deploy/*.sh`、core、converter、connector 源码。
  - 当前 `examples/binance_tick_mm/deploy/run_live.sh` 的 dirty 状态来自此前 `0513T004`，不是本任务修改。

done：
- 生成全部 required artifacts：
  - `sample_inventory.json`
  - `latency_summary.json`
  - `latency_summary.csv`
  - `market_data_integrity.json`
  - `top5_mismatch_buckets.csv`
  - `source_provenance_summary.json`
  - `sample_usability_matrix.csv`
  - `step2_baseline_report.md`
- 样本 classification：
  - `5-13-day-control-15min`: `pricing_research_candidate`
  - `5-11-night-active`: `compressed_action_path_only`
  - `5-10-day-control-1h-06`: `compressed_action_path_only`
  - `5-9-noon`: `compressed_action_path_only`
  - `5-9-small`: `compressed_action_path_only`
- 主要 latency / data-quality 结论：
  - 5 个样本现有 maker action-path acceptance 均可用，但这仍是 compressed action-path gate，不是 full L2 / queue / OFI / microprice proof。
  - 只有 `5-13-day-control-15min` 有 T002 MarketView provenance 字段；其他旧样本缺少 strategy-layer source fields。
  - 5 个样本均无 T004 `deployment_manifest.json`，从部署可追溯角度都属于 legacy/pre-T004 样本。
  - 所有样本 order-entry tail latency 都是 degraded；entry p99 约 `4055ms` 到 `15112ms`。
  - raw gzip depth `pu` continuity 在本轮 bounded scan 中均为 `0` mismatch。
  - bookTicker/depth consistency 只是 bounded best-effort 检查，每个样本检查 `20000` 条，不是 production-grade local book proof。
  - top5 tick/qty mismatch 仍然 material：tick match 约 `0.9044` 到 `0.9747`，qty match 约 `0.8898` 到 `0.9558`。
  - converted npz 不保留 Binance `U/u/pu`、`lastUpdateId`、bookTicker provenance。
- 需要后续 core/data task 的字段：
  - Binance depth `U/u/pu`
  - snapshot `lastUpdateId`
  - bookTicker provenance
  - per-decision top-N book snapshots
  - per-decision raw/update-id provenance join
- 本任务是 read-only baseline 实施，不是 live promotion，不授权策略变更、pricing/queue 研究实现、core/data schema 修改或 live micro test。

blockers：
- 无。

commit：
- 无

提交信息：
- 无
```
