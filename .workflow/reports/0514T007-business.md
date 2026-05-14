```md
执行线程：
- 业务线程-python

任务ID：
- 0514T007

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0514T007.md`
- `.workflow/reports/0514T007-business.md`
- `examples/binance_tick_mm/execution_outcome_calibration.py`
- `examples/binance_tick_mm/test_execution_outcome_calibration.py`
- `task_plan.md`
- `progress.md`
- `findings.md`
- `.workflow/dashboard.html`
- `.workflow/dispatch_suggestions.md`
- `local_live_analysis/5-13-day-control-30min/stage6_execution_calibration_0514T007/**`

action：
- 新增 `examples/binance_tick_mm/execution_outcome_calibration.py` 只读 Stage 6B runner：
  - 复用 Stage 5 `build_execution_labels(...)` 在 live / replay 两侧生成同 schema execution rows。
  - 用 `submit_strategy_seq + order_side` 构造 normalized `submit_key`，作为 matched submit opportunity comparison unit。
  - 显式输出 `submit_key_coverage.csv`，区分 matched / unmatched coverage。
  - 生成 aggregate calibration gaps：
    - fill horizon
    - time-to-fill
    - final order state
    - cancel race
    - markout / spread-retention
    - coverage / observability gap
  - 生成 strata calibration gaps：
    - placement
    - inventory / size proxy
    - latency / join-age / stale
  - 输出 decision state：`methodology_valid_single_sample` / `diagnostic_only_gap_too_large` / `requires_more_current_format_samples`
- 新增 `examples/binance_tick_mm/test_execution_outcome_calibration.py`：
  - 覆盖 required artifacts
  - 覆盖 matched submit key coverage
  - 覆盖 fill-gap / cancel-race gap / strata output
- 在 `5-13-day-control-30min` 上全量运行 Stage 6B，产物输出到：
  - `local_live_analysis/5-13-day-control-30min/stage6_execution_calibration_0514T007/`

run result：
- dataset：`5-13-day-control-30min`
- stage3 classification used：`passes_pricing_research_market_view`
- decision state：`diagnostic_only_gap_too_large`
- submit matching：
  - live submit orders：`2516`
  - replay submit orders：`2516`
  - matched submit orders：`2516`
  - matched coverage vs live：`1.0`
  - matched coverage vs replay：`1.0`
  - price tick equality on matched submits：`2516/2516`
  - qty equality on matched submits：`2516/2516`
- lifecycle counts：
  - live filled orders：`53`
  - replay filled orders：`172`
  - live fill-after-cancel orders：`16`
  - replay fill-after-cancel orders：`133`

what aligned：
- submit-key coverage aligned：
  - normalized matched submit opportunity coverage 是 `1.0`
  - matched submits 的 price tick / qty equality 是 `2516/2516`
- fill probability 大体对齐：
  - `100ms` gap：`0.00119`
  - `500ms` gap：`0.00119`
  - `1000ms` gap：`0.00199`
  - `5000ms` gap：`0.01192`
  - 按当前 Stage 6A 单样本阈值，这四个 horizon 都在 aligned bucket 内
- fast-cancel-churn aligned：
  - live `0.7770`
  - replay `0.7770`
  - gap `0.0`
- submit observability coverage aligned：
  - fill_probability observable coverage 在 `100/500/1000/5000ms` 上 live / replay 一致

what not aligned：
- final order state 不对齐：
  - `canceled` gap：`0.09062`
  - `filled` gap：`0.04730`
  - `open_or_missing` gap：`0.04332`
- cancel-to-fill race 不对齐：
  - fill-after-cancel-request rate：
    - live `0.00636`
    - replay `0.05286`
    - gap `0.04650`
  - cancel-to-fill delay mean：
    - live observed `13.90ms`
    - replay observed `30977.86ms`
    - large aggregate mismatch
- markout observability coverage 不对齐：
  - `100ms` fill markout coverage gap：`0.03816`
  - `500ms` gap：`0.03617`
  - `1000ms` gap：`0.02941`
  - `5000ms` gap：`0.02901`
- replay fills are materially denser than live：
  - live fills `53`
  - replay fills `172`
  - 说明 replay lifecycle / fill model 仍显著更积极

where gaps concentrate：
- placement strata：
  - `step_back_gt1` 在 `5000ms` fill rate 上 gap `0.01360`
  - `touch` 在 fill-after-cancel rate 上 gap `0.00973`
- inventory strata：
  - `inventory_score q5` 在 `5000ms` fill rate gap `0.01468`
  - `same_side_top1_qty q5` fill-after-cancel gap `0.01984`
- latency / join-age strata：
  - `latency_signal_ms q5` fill-after-cancel gap `0.01386`
  - `top5_join_age_ms q5` `5000ms` fill gap `0.01587`
  - `join_stale=0` fill-after-cancel gap `0.04657`
- aggregate-only 报告会低估这些 strata-level 差异，因此 Stage 6A 的 strata requirement 是必要的

interpretation boundary：
- 本 runner 只比较 observed replay/live lifecycle proxies。
- `submit_key` 匹配是为比较同一 submit opportunity，不是 counterfactual proof。
- queue priority、missed opportunity、realized PnL decomposition 仍然只是 observed-only proxy 语义；本任务没有把它们升级成 exact queue / counterfactual fill 结论。
- 单样本 `diagnostic_only_gap_too_large` 只能说明 replay fill/cancel lifecycle 仍偏离 live，不构成 strategy PnL proof 或 live readiness proof。

verify：
- `python -m pytest examples/binance_tick_mm/test_execution_outcome_calibration.py` -> `3 passed`
- `python examples/binance_tick_mm/execution_outcome_calibration.py --help` -> passed
- `python examples/binance_tick_mm/execution_outcome_calibration.py --run-dir local_live_analysis/5-13-day-control-30min --output-dir local_live_analysis/5-13-day-control-30min/stage6_execution_calibration_0514T007` -> passed
- 人工检查输出目录包含 required artifacts -> passed
- `python3 .workflow/build_dashboard.py` -> passed
- `git diff --check` -> passed

done：
- `0514T007` 已实现 read-only Stage 6B calibration runner，并在 `5-13-day-control-30min` 上全量运行。
- required outputs 已生成到 `local_live_analysis/5-13-day-control-30min/stage6_execution_calibration_0514T007/`。
- 结果表明：
  - matched submit coverage 对齐
  - fill horizon 大体对齐
  - 但 final state、fill-after-cancel-request、cancel-to-fill delay、markout observability coverage 仍未对齐
  - current decision state 应为 `diagnostic_only_gap_too_large`
- 本任务未改策略、未启动 live、未跑新 replay sweep、未改 core/connector/schema。

blockers：
- 无

commit：
- 待提交

提交信息：
- 待提交
```
