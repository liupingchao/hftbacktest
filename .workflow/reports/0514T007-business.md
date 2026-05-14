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
- `local_live_analysis/5-13-day-control-30min/stage6_execution_calibration_0514T007/execution_calibration_summary.md`
- `local_live_analysis/5-13-day-control-30min/stage6_execution_calibration_0514T007/submit_key_coverage.csv`
- `local_live_analysis/5-13-day-control-30min/stage6_execution_calibration_0514T007/fill_horizon_gap.csv`
- `local_live_analysis/5-13-day-control-30min/stage6_execution_calibration_0514T007/time_to_fill_gap.csv`
- `local_live_analysis/5-13-day-control-30min/stage6_execution_calibration_0514T007/final_state_gap.csv`
- `local_live_analysis/5-13-day-control-30min/stage6_execution_calibration_0514T007/cancel_race_gap.csv`
- `local_live_analysis/5-13-day-control-30min/stage6_execution_calibration_0514T007/markout_gap.csv`
- `local_live_analysis/5-13-day-control-30min/stage6_execution_calibration_0514T007/placement_strata_gap.csv`
- `local_live_analysis/5-13-day-control-30min/stage6_execution_calibration_0514T007/inventory_strata_gap.csv`
- `local_live_analysis/5-13-day-control-30min/stage6_execution_calibration_0514T007/latency_strata_gap.csv`
- `local_live_analysis/5-13-day-control-30min/stage6_execution_calibration_0514T007/coverage_gap.csv`
- `local_live_analysis/5-13-day-control-30min/stage6_execution_calibration_0514T007/run_manifest.json`

action：
- 新增 `examples/binance_tick_mm/execution_outcome_calibration.py` 只读 Stage 6B runner：
  - 复用 Stage 5 `build_execution_labels(...)` 口径分别构造 live 与 audit replay 的 execution rows。
  - 用 `submit_strategy_seq + order_side` 作为 normalized submit key，在当前样本上构造 matched submit opportunities。
  - 输出 submit-key coverage、fill horizon gap、time-to-fill gap、final-state gap、cancel-race gap、markout gap、placement/inventory/latency strata gaps。
  - 用 single-sample decision state 生成 summary markdown 和 manifest。
- 新增 `examples/binance_tick_mm/test_execution_outcome_calibration.py`：
  - 覆盖 required artifacts 输出。
  - 覆盖 submit-key matching。
  - 覆盖 cancel race gap 与 strata family 输出。
- 在 `5-13-day-control-30min` 上全量运行 Stage 6B runner，输出到：
  - `local_live_analysis/5-13-day-control-30min/stage6_execution_calibration_0514T007/`

run result：
- dataset：`5-13-day-control-30min`
- stage3 classification：`passes_pricing_research_market_view`
- decision_state：`diagnostic_only_gap_too_large`
- submit coverage：
  - live submit orders：`2516`
  - replay submit orders：`2516`
  - matched submit orders：`2516`
  - matched coverage vs live：`1.0`
  - matched coverage vs replay：`1.0`
  - matched price tick equality：`2516/2516`
  - matched qty equality：`2516/2516`
- lifecycle counts：
  - live filled orders：`53`
  - replay filled orders：`172`
  - live fill-after-cancel orders：`16`
  - replay fill-after-cancel orders：`133`

what aligned：
- comparison-unit coverage：
  - normalized submit-key coverage 在当前样本上是完整的，`2516/2516` matched。
- short-horizon fill rates：
  - `100ms` fill gap：`0.0012`
  - `500ms` fill gap：`0.0012`
  - `1000ms` fill gap：`0.0020`
- `fast_cancel_churn_rate`：
  - gap：`0.0000`

what did not align：
- final order state distribution：
  - `canceled` gap：`0.0906`
  - `filled` gap：`0.0473`
  - `open_or_missing` gap：`0.0433`
- cancel race：
  - `fill_after_cancel_request_rate`：live `0.00636` vs replay `0.05286`
  - absolute gap：`0.04650`
  - cancel-to-fill delay mean：live `13.90ms` vs replay `30977.86ms` on all observed rows
  - matched both-observed cancel-to-fill delay mean gap 仍有 `98.78ms`
- long-horizon fill：
  - `5000ms` fill gap：`0.0119`
- time-to-fill：
  - all-filled mean：live `2684.94ms` vs replay `25176.57ms`
  - all-filled p50：live `884.26ms` vs replay `8309.94ms`
  - all-filled p90：live `8783.07ms` vs replay `59352.75ms`
  - even on matched-both-filled rows，mean gap 仍有 `370.19ms`

where gaps concentrate：
- placement / edge：
  - `distance_to_bbo_ticks_bucket=q3` 的 cancel-fill gap 约 `0.0631`
  - `edge_vs_fair_ticks_bucket=q2` 的 cancel-fill gap 约 `0.0669`
  - `placement_bucket=step_back_gt1` 的 cancel-fill gap 约 `0.0564`
- inventory / top-size proxy：
  - `same_side_top1_qty_bucket=q3` 的 cancel-fill gap 约 `0.0676`
  - `inventory_score_bucket=q4` 的 cancel-fill gap 约 `0.0570`
- latency / join-age：
  - `top5_join_age_ms_bucket=q5` 的 cancel-fill gap 约 `0.0615`
  - `latency_signal_ms_bucket=q4/q5` 的 cancel-fill gap 约 `0.0515-0.0518`
  - 某些 latency buckets 的 time-to-fill mean gap 仍很大，例如 `latency_signal_ms_bucket=q2` 约 `3970ms`

boundary：
- 未修改 strategy behavior、fair/target 公式、配置默认值、risk guards 或 quote placement。
- 未启动 live、未新跑 replay sweep、未改 AWS/remote state。
- 未修改 core Rust、connector、py `event_dtype`、canonical `audit_schema.py` 或标准 npz schema。
- Stage 6B 结果只是 observed replay/live lifecycle proxy calibration：
  - 不能解释为 exact queue proof。
  - 不能解释为 counterfactual fill proof。
  - 不能解释为策略 PnL proof 或 live readiness。
- 当前单样本结果只说明 methodology 可运行且差异可定位；它不授权 quote-adjustment promotion。

verify：
- `python -m pytest examples/binance_tick_mm/test_execution_outcome_calibration.py` -> `3 passed`
- `python examples/binance_tick_mm/execution_outcome_calibration.py --help` -> passed
- `python examples/binance_tick_mm/execution_outcome_calibration.py --run-dir local_live_analysis/5-13-day-control-30min --output-dir local_live_analysis/5-13-day-control-30min/stage6_execution_calibration_0514T007` -> passed
- 人工检查输出目录包含 required artifacts -> passed
- `python3 .workflow/build_dashboard.py` -> passed
- `git diff --check` -> passed

done：
- `0514T007` 已实现 read-only calibration runner 和 focused tests。
- 已在 `5-13-day-control-30min` 上完成 single-sample Stage 6B 运行。
- 主结论是：
  - submit-key coverage 完整
  - 短 horizon fill rate 与 fast-cancel churn 大致对齐
  - 但 final state、fill-after-cancel-request、long-horizon fill 和 time-to-fill 仍明显不对齐
  - 差异集中在 placement / edge / inventory / latency 某些 buckets
- 当前 decision_state 为 `diagnostic_only_gap_too_large`。

blockers：
- 无执行阻塞。
- 研究阻塞仍存在：当前单样本 gap 过大且 fills 稀疏，不足以支持 quote-adjustment promotion。

commit：
- 待提交

提交信息：
- 待提交
```
