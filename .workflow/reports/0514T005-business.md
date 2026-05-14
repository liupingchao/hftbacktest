```md
执行线程：
- 业务线程-python

任务ID：
- 0514T005

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0514T005.md`
- `.workflow/reports/0514T005-business.md`
- `examples/binance_tick_mm/execution_outcome_labels.py`
- `examples/binance_tick_mm/test_execution_outcome_labels.py`
- `task_plan.md`
- `progress.md`
- `findings.md`
- `.workflow/dashboard.html`
- `.workflow/dispatch_suggestions.md`
- `local_live_analysis/5-13-day-control-30min/stage5_execution_outcome_labels_0514T005/execution_outcome_label_summary.md`
- `local_live_analysis/5-13-day-control-30min/stage5_execution_outcome_labels_0514T005/execution_outcome_labels.csv`
- `local_live_analysis/5-13-day-control-30min/stage5_execution_outcome_labels_0514T005/label_coverage.csv`
- `local_live_analysis/5-13-day-control-30min/stage5_execution_outcome_labels_0514T005/label_statistics.csv`
- `local_live_analysis/5-13-day-control-30min/stage5_execution_outcome_labels_0514T005/fill_horizon_labels.csv`
- `local_live_analysis/5-13-day-control-30min/stage5_execution_outcome_labels_0514T005/fill_markout_labels.csv`
- `local_live_analysis/5-13-day-control-30min/stage5_execution_outcome_labels_0514T005/placement_opportunity_tradeoff.csv`
- `local_live_analysis/5-13-day-control-30min/stage5_execution_outcome_labels_0514T005/lifecycle_label_summary.csv`
- `local_live_analysis/5-13-day-control-30min/stage5_execution_outcome_labels_0514T005/censoring_summary.csv`
- `local_live_analysis/5-13-day-control-30min/stage5_execution_outcome_labels_0514T005/run_manifest.json`

action：
- 新增 `examples/binance_tick_mm/execution_outcome_labels.py` 只读 runner：
  - 从 live audit 的真实 `order_submit_sent / cancel_sent / cancel_ack / fill / expired` 生命周期行构造订单级 execution outcome labels。
  - 用 T009 `joined_decisions.csv` 的 decision context 为 submit 行补齐 top5、latency、inventory、target/working tick、join age 等观测上下文。
  - 生成 `100ms / 500ms / 1s / 5s` 的 fill horizon、fill markout、missed opportunity proxy、censoring 和 inventory-cycle labels。
  - 明确把 `queue_priority_proxy`、`missed_fill_opportunity_cost`、`realized_pnl_decomposition`、`post_only_reject_throttle_churn` 标成 observed-only proxy，而不是 counterfactual / exact queue 结论。
  - 输出 required artifacts，并在 `label_coverage.csv` 中对每个 label class 给出 `available / observed_only_proxy / low_sample` 状态和原因。
- 新增 `examples/binance_tick_mm/test_execution_outcome_labels.py`：
  - 覆盖真实 submit/fill 订单。
  - 覆盖 cancel-after-fill race。
  - 覆盖未成交订单的 censoring 和 missed-opportunity proxy。
  - 覆盖 required artifacts 和 coverage status。
- 在 `5-13-day-control-30min` 上全量运行 Stage 5 label runner，产物输出到：
  - `local_live_analysis/5-13-day-control-30min/stage5_execution_outcome_labels_0514T005/`

run result：
- dataset：`5-13-day-control-30min`
- stage3 classification used：`passes_pricing_research_market_view`
- submit orders：`2516`
- filled orders：`53`
- fill-after-cancel orders：`16`
- partial-fill orders：`0`
- missing lifecycle orders：`2`
- fill-by-horizon：
  - `100ms`: `8`
  - `500ms`: `22`
  - `1000ms`: `28`
  - `5000ms`: `40`
- fill horizon observable rows：
  - `100ms`: `2515`
  - `500ms`: `2515`
  - `1000ms`: `2515`
  - `5000ms`: `2514`
- fill markout observable rows：
  - `100ms`: `44`
  - `500ms`: `34`
  - `1000ms`: `33`
  - `5000ms`: `35`
- lifecycle summary：
  - `canceled`: `2452`
  - `filled`: `53`
  - `expired`: `9`
  - `open_or_missing`: `2`

T004 label coverage on `5-13-day-control-30min`：
- fully observed / available：
  - `fill_probability`
  - `time_to_fill`
  - `adverse_selection_after_fill`
  - `spread_capture`
  - `cancel_to_fill_race`
  - `inventory_impact`
  - `quote_placement_distance`
  - `partial_fill_lifecycle`
  - `inventory_cycle`
  - `sample_validity_censoring`
- observed-only proxy：
  - `queue_priority_proxy`
  - `post_only_reject_throttle_churn`
  - `missed_fill_opportunity_cost`
  - `realized_pnl_decomposition`
- low sample：
  - `tail_risk`：当前只有 `53` 个 filled orders，tail quantile / tail mean / worst-bucket concentration 已实现并输出，但仍应按 low-sample 解释。
- unavailable：
  - 无。T004 要求的 label class 在当前数据集上都已实现；不能 fully observe 的部分已显式降级为 observed-only proxy。

implemented statistics：
- continuous：Pearson、Spearman、5-bucket mean/median、top-bottom spread、monotonicity、time-split stability。
- binary：event rate、lift vs baseline、top-vs-bottom odds ratio。
- time-to-event / censored：horizon-level censoring counts、discrete hazard、survival summary；summary 中明确 Cox-style model 仍是后续建模候选。
- count/rate：count、exposure-normalized rate、top-bottom rate ratio。
- multiclass/lifecycle：placement bucket x final order state conditional probability / one-vs-rest lift。
- tail：p1/p5 proxy、tail mean、exceedance rate、worst placement-bucket concentration。
- placement / opportunity-cost：`placement_bucket x signal_bucket` tradeoff table。

boundary：
- 未修改 strategy behavior、fair/target 公式、配置默认值、risk guards 或 quote placement。
- 未启动 live、未重新采集、未改 AWS/remote state。
- 未修改 core Rust、connector、py `event_dtype`、canonical `audit_schema.py` 或标准 npz schema。
- 未覆盖历史 Stage 4 / T009 / T007 产物目录。
- observed labels、proxy labels、相关性摘要都不应解释成 counterfactual queue/fill proof、策略 PnL 证明或 live readiness 证明。

verify：
- `python -m pytest examples/binance_tick_mm/test_execution_outcome_labels.py` -> `3 passed`
- `python examples/binance_tick_mm/execution_outcome_labels.py --help` -> passed
- `python examples/binance_tick_mm/execution_outcome_labels.py --run-dir local_live_analysis/5-13-day-control-30min --output-dir local_live_analysis/5-13-day-control-30min/stage5_execution_outcome_labels_0514T005` -> passed
- 人工检查输出目录包含全部 required artifacts -> passed
- 人工检查 `label_coverage.csv` 覆盖 T004 原始 outcome 主线和新增 7 类 label，且每类 label 有状态与原因 -> passed
- 人工检查 `label_statistics.csv` 覆盖 `14` 个 label classes、`7` 种统计类型 -> passed
- `python3 .workflow/build_dashboard.py` -> passed
- `git diff --check` -> passed

done：
- `0514T005` 已实现 read-only execution outcome label runner，并在 `5-13-day-control-30min` 上全量运行。
- required outputs 已生成到 `local_live_analysis/5-13-day-control-30min/stage5_execution_outcome_labels_0514T005/`。
- T004 要求的 label class 全部已在当前数据集上实现并测试；不能 fully observe 的部分已显式标注 observed-only proxy 或 low-sample。
- 当前结果进入 QA。

blockers：
- 无

commit：
- da9244e

提交信息：
- feat(binance): add stage5 execution outcome labels
```
