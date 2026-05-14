```md
执行线程：
- 业务线程-python

任务ID：
- 0514T006

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0514T006.md`
- `.workflow/reports/0514T006-business.md`
- `task_plan.md`
- `progress.md`
- `findings.md`
- `.workflow/dashboard.html`
- `.workflow/dispatch_suggestions.md`

action：
- 读取并整理 Stage 6A planning inputs：
  - `0514T005` task / business / QA
  - `5-13-day-control-30min` Stage 5 execution-outcome summary
  - `docs/stage6i-cancel-requested-fill-risk-plan.md`
- 定义 Stage 6 refined objective：
  - 从宽泛 queue-model 语言，收窄为 replay/live fill-cancel lifecycle proxy calibration。
  - 目标是让后续 quote-adjustment PnL 研究中的 fill-side假设更可信，而不是宣称 exact queue proof。
- 明确为什么要微调：
  - T005 当前样本 shape 是 high cancel / low fill / non-trivial cancel race。
  - `queue_priority_proxy`、`missed_fill_opportunity_cost`、`realized_pnl_decomposition` 仍是 observed-only proxy。
  - fills 只有 `53`，partial fills 为 `0`，tail risk 仍是 low-sample。
- 定义 Stage 6 comparison unit：
  - 跨 live/replay 对齐不能依赖 raw `order_id`。
  - 应以 matched normalized submit opportunities 为主 comparison unit，并单独报告 matched / unmatched coverage。
- 定义 Stage 6 common label schema：
  - fill by `100ms / 500ms / 1s / 5s`
  - time-to-fill
  - final order state / lifecycle
  - fill-after-cancel-request
  - cancel-to-fill delay
  - fast-cancel-churn
  - fill markout / spread-retention
  - coverage / censoring flags
- 定义必做 strata：
  - `placement_bucket`
  - `distance_to_bbo_ticks`
  - `edge_vs_fair_ticks`
  - `inventory_score`
  - top-of-book / top5 size-age proxy
  - latency regime
- 定义 acceptance / decision states：
  - `methodology_valid_single_sample`
  - `diagnostic_only_gap_too_large`
  - `requires_more_current_format_samples`
  - later-only `limited_offline_quote_experiment_candidate`
- 明确样本策略：
  - `5-13-day-control-30min` 足够做单样本方法论验证与 runner shakeout。
  - 但它不足以单独授权 quote-adjustment promotion，需要更多 current-format windows。
- 明确 `0514T007` 边界：
  - 只能做 read-only calibration runner。
  - 不启动 live，不生成新 replay sweep，不改策略，不做 exact queue / counterfactual fill 结论。

Stage 6A contract：

1. Objective
   - 判断 replay/live 的 execution-outcome proxy 是否已经足够接近，以支持后续 limited quote-adjustment PnL experiments。
   - Stage 6 关注的是 lifecycle proxy error，而不是 exact queue reconstruction。

2. Non-goals
   - 不做 exact MBO queue reconstruction。
   - 不做 counterfactual fill simulation。
   - 不做策略规则、inventory logic、quote-update logic 实现。
   - 不做 live promotion 或 live micro test 判定。

3. Comparison unit
   - `0514T007` 必须先构造 matched submit opportunity coverage。
   - 应分别报告：
     - live submit count
     - replay submit count
     - matched submit count
     - unmatched live submit count
     - unmatched replay submit count
   - 主要 lifecycle / fill-gap 结论必须基于 matched submit universe，并把 coverage gap 单列。

4. Required labels
   - binary / event:
     - `fill_by_100ms`
     - `fill_by_500ms`
     - `fill_by_1000ms`
     - `fill_by_5000ms`
     - `fill_after_cancel_request`
     - `fast_cancel_churn`
   - continuous / time:
     - `time_to_fill_ms`
     - `cancel_to_fill_delay_ms`
     - `fill_markout_{100,500,1000,5000}ms_ticks`
     - `realized_spread_proxy_ticks`
   - lifecycle:
     - `final_order_state`
     - `full_fill / partial_fill / no_fill`
   - coverage:
     - `horizon_observable`
     - `right_censored`
     - `tail_truncated`
     - `missing_lifecycle`

5. Required strata
   - placement:
     - `placement_bucket`
     - `distance_to_bbo_ticks`
   - pricing state:
     - `edge_vs_fair_ticks`
   - inventory state:
     - `inventory_score`
   - book / queue proxy:
     - same-side / opposite-side top1-top5 size bucket
     - `top5_join_age_ms`
     - `join_stale`
   - latency:
     - `latency_signal_ms` bucket
     - freshness / stale-age bucket if available

6. Required metrics in `0514T007`
   - coverage:
     - matched submit coverage
     - per-label observable / censored / truncated counts
   - aggregate gaps:
     - fill-horizon absolute gap and relative gap
     - time-to-fill p50 / p90 / mean gap
     - final-state distribution gap
     - fill-after-cancel rate gap
     - cancel-to-fill latency p50 / p90 gap
     - fill-markout mean / median gap
   - strata gaps:
     - the same metrics by placement / edge / inventory / latency / top5-age strata
   - sample-validity notes:
     - low-fill or sparse-strata warnings
     - unavailable comparisons caused by zero matched rows

7. Decision states for Stage 6B output
   - `methodology_valid_single_sample`
   - `diagnostic_only_gap_too_large`
   - `requires_more_current_format_samples`
   - `limited_offline_quote_experiment_candidate` only as a later multi-sample controller decision

8. Sample strategy
   - `5-13-day-control-30min`:
     - enough for Stage 6B runner implementation and single-sample methodology validation
     - not enough alone to authorize quote-adjustment PnL experiment promotion
   - later multi-sample extension:
     - require additional current-format windows with the same Stage 3 and Stage 5 artifact chain
     - recommended regimes: one quieter/churn-normal window and one directional/high-churn window

9. Recommended `0514T007` outputs
   - `execution_calibration_summary.md`
   - `submit_key_coverage.csv`
   - `fill_horizon_gap.csv`
   - `time_to_fill_gap.csv`
   - `final_state_gap.csv`
   - `cancel_race_gap.csv`
   - `markout_gap.csv`
   - `placement_strata_gap.csv`
   - `inventory_strata_gap.csv`
   - `latency_strata_gap.csv`
   - `coverage_gap.csv`
   - `run_manifest.json`

10. Recommended next task
    - `0514T007` should implement the read-only Stage 6B runner on `5-13-day-control-30min`.
    - 它必须显式报告：哪些 gap 是 lifecycle mismatch，哪些只是 coverage mismatch，哪些结论仍是 observed-only proxy。

verify：
- 人工检查 `.workflow/tasks/0514T006.md` 与 `.workflow/reports/0514T006-business.md` 是否完整。
- `python3 .workflow/build_dashboard.py`
- `git diff --check`

done：
- Stage 6A planning contract 已形成。
- 已明确 Stage 6 为什么从 queue-model 语言微调为 fill/cancel lifecycle proxy calibration。
- 已明确 Stage 6 common labels、comparison unit、分层维度、acceptance metrics、样本要求和 `0514T007` 实现边界。
- 已明确 `5-13-day-control-30min` 仅足够做单样本方法论验证，不足以单独授权 quote-adjustment promotion。
- 本任务未实现 runner、未运行 replay、未启动 live、未改策略、未改 core/connector/schema。

blockers：
- 无

commit：
- 待提交

提交信息：
- 待提交
```
