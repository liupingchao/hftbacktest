```md
执行线程：
- 业务线程-python

任务ID：
- 0514T001

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0514T001.md`
- `.workflow/reports/0514T001-business.md`
- `task_plan.md`
- `progress.md`
- `findings.md`
- `.workflow/dashboard.html`
- `.workflow/dispatch_suggestions.md`
- `examples/binance_tick_mm/maker_acceptance.py`
- `examples/binance_tick_mm/test_maker_acceptance.py`
- `local_live_analysis/5-13-day-control-30min/maker_acceptance_stage3.json`

action：
- 判断 `5-13-day-control-30min` 是否足够支撑 Stage 3 验收：
  - 足够。该样本已有 T004 preflight/manifest/start-stop marker、action-path maker acceptance、T009 fixed sidecar metrics、T009 joined-decision metrics、sidecar/join provenance CSV、以及已知 live audit top5 vs sidecar/replay top5 非完全一致的质量指标。
  - 不需要重新 live 采集，不需要继续修 T009 bootstrap，也不需要单独 planning-only task。
- 扩展 `examples/binance_tick_mm/maker_acceptance.py`：
  - 保留原 action/planned/reject/throttle/working-order/replay-lag hard gates。
  - 新增可选 `--sidecar-metrics`、`--joined-decision-metrics`、`--top5-sidecar-csv`、`--joined-decisions-csv`。
  - 新增 `market_view` section，包含 required gates、quality gates、derived metrics、thresholds、top5 diagnostics、classification 和边界说明。
  - 默认不传 sidecar/join 参数时，`market_view.enabled=false`，旧行为保持不变。
- 新增 focused tests：
  - 旧 maker acceptance 行为不回退。
  - T009 sidecar/join metrics 通过时分类为 `passes_pricing_research_market_view`。
  - `future_join_count` / `gap_crossed_join_count` 非零时降级为 `compressed_action_path_only`。
  - stale/top5 quality 未过阈值时分类为 `limited_pricing_research`。
  - 原 action-path hard gate 失败时分类为 `unusable`。
- 使用完整 `5-13-day-control-30min` 运行 Stage 3 acceptance，输出 `local_live_analysis/5-13-day-control-30min/maker_acceptance_stage3.json`。

verify：
- `python -m pytest examples/binance_tick_mm/test_maker_acceptance.py` -> `6 passed`
- `python examples/binance_tick_mm/maker_acceptance.py --alignment-report local_live_analysis/5-13-day-control-30min/alignment_report_audit_replay.json --backtest-result local_live_analysis/5-13-day-control-30min/backtest_audit_replay_result.json --sidecar-metrics local_live_analysis/5-13-day-control-30min/t009_fixed_sidecar/metrics.json --joined-decision-metrics local_live_analysis/5-13-day-control-30min/t009_fixed_sidecar/joined_decisions.metrics.json --top5-sidecar-csv local_live_analysis/5-13-day-control-30min/t009_fixed_sidecar/top5_sidecar.csv --joined-decisions-csv local_live_analysis/5-13-day-control-30min/t009_fixed_sidecar/joined_decisions.csv --out local_live_analysis/5-13-day-control-30min/maker_acceptance_stage3.json` -> exit `0`
- Stage 3 full-run result：
  - `passed=true`
  - `market_view.enabled=true`
  - `market_view.classification=passes_pricing_research_market_view`
  - hard failures：`[]`
- Required gates：
  - `first_valid_update_aligned=true`
  - `depth_pu_mismatch_count=0`
  - `final_data_row_mapping_coverage=1.0`
  - `decision_join_coverage=1.0`
  - `future_join_count=0`
  - `join_missing_count=0`
  - `gap_crossed_join_count=0`
- Quality gates：
  - bookTicker/depth BBO mismatch rate：`0.00016340354734246414`，threshold `<=0.001`
  - stale join rate：`0.0090949283142803`，threshold `<=0.02`
  - top5 join age p99：`28.13446387999999ms`，threshold `<=50ms`
  - best bid tick match rate：`0.8236483072258717`，threshold `>=0.80`
  - best ask tick match rate：`0.8236904160350346`，threshold `>=0.80`
  - top5 tick match rate：`0.8232061647296615`，threshold `>=0.80`
  - top5 qty match rate：`0.8014359103924541`，threshold `>=0.75`
- `python3 .workflow/build_dashboard.py` -> dashboard refreshed

done：
- Stage 3 market-view acceptance gate 已实现并通过 focused tests。
- `5-13-day-control-30min` 满足本任务验收需要，并在 Stage 3 中分类为 `passes_pricing_research_market_view`，可作为后续 top5 microprice / top5 OFI proxy / top5 imbalance pricing research 的 market-view candidate。
- top5 tick/qty 仍不是逐行完全一致；Stage 3 将其作为阈值化 quality gate 和 diagnostics，而不是 full L2 / exact queue proof。
- 没有启动 live，没有重新采集，没有策略行为变更，没有 core/connector/API 变更，没有 standard npz schema 变更。

blockers：
- 无

commit：
- 7320da4

提交信息：
- feat(binance): add market-view acceptance gate
```
