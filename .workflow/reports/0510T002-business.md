```md
执行线程：
- 测试线程

任务ID：
- 0510T002

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- .workflow/tasks/0510T002.md
- .workflow/runners/run_task.py
- .workflow/runners/run_0510T002.py
- examples/binance_tick_mm/stage6j_replay.py
- local_live_analysis/stage6j_cross_sample_0510T002/*

action：
- 自动检查跨样本输入是否存在。
- 自动补算缺失的 `maker_acceptance.json`。
- 自动执行 Stage 6J 跨样本 replay。
- 自动解析 `stage6j_replay_decision.json` 和 `stage6j_replay_summary.csv`。
- 自动生成本执行回报。
- 自动刷新 workflow dashboard。
- 未启动真实 live，未连接交易所，未改 AWS 状态。

verify：
- 命令：`/home/molly/anaconda3/bin/python3 examples/binance_tick_mm/stage6j_replay.py --local-root local_live_analysis --out-dir local_live_analysis/stage6j_cross_sample_0510T002 --run-id 5-10-day-control-1h-06 --run-id 5-9-small --run-id 5-9-noon --run-id 5-8-stage3-15m-livetest-v4`
- exit code：`0`
- 输出目录：`local_live_analysis/stage6j_cross_sample_0510T002`
- 样本列表：
- 5-10-day-control-1h-06
- 5-9-small
- 5-9-noon
- 5-8-stage3-15m-livetest-v4
- 缺失输入：
- 无
- preflight：
- 5-10-day-control-1h-06: maker_acceptance.json exists
- 5-9-small: maker_acceptance.json exists
- 5-9-noon: maker_acceptance.json exists
- 5-8-stage3-15m-livetest-v4: generated maker_acceptance.json

done：
- Stage 6J decision：`diagnostic_only_no_promotion`
- 样本数：`4`
- 候选数：`6`
- hard failures：`0`
- 候选汇总：
- add_side_guard_only: runs=4, pnl_sum=-0.232900, max_abs_notional=242.439450, cancel_fill=0, inv_worsening_no_readd=0, same_side_worsening=0
- add_side_guard_post_fill_100ms: runs=4, pnl_sum=-0.232900, max_abs_notional=242.439450, cancel_fill=0, inv_worsening_no_readd=0, same_side_worsening=0
- add_side_guard_post_fill_200ms: runs=4, pnl_sum=-0.232900, max_abs_notional=242.439450, cancel_fill=0, inv_worsening_no_readd=0, same_side_worsening=0
- add_side_guard_post_fill_50ms: runs=4, pnl_sum=-0.232900, max_abs_notional=242.439450, cancel_fill=0, inv_worsening_no_readd=0, same_side_worsening=0
- baseline_inflight_only: runs=4, pnl_sum=-0.864100, max_abs_notional=242.439450, cancel_fill=3, inv_worsening_no_readd=0, same_side_worsening=3
- broad_add_side_cooldown_200ms_control: runs=4, pnl_sum=-0.053500, max_abs_notional=242.439450, cancel_fill=1, inv_worsening_no_readd=0, same_side_worsening=1
- 下一步建议：保持 diagnostic-only，不自动进入 live micro test。总控应人工比较跨样本 PnL、max position、drop rate、churn 和 cancel-fill source-path，再决定是否创建 adverse-selection timing rule 设计任务。
- runner stdout 摘要：
```text
{
  "decision": "diagnostic_only_no_promotion",
  "samples": 4,
  "candidates": 6,
  "hard_failures": 0
}
```

blockers：
- 无

commit：
- 无

提交信息：
- 无
```
