```md
执行线程：
- 业务线程-python

任务ID：
- 0518T003

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0518T003.md`
- `.workflow/reports/0518T003-business.md`
- `examples/binance_tick_mm/quote_anchor_diagnostic.py`
- `examples/binance_tick_mm/test_quote_anchor_diagnostic.py`
- `task_plan.md`
- `progress.md`
- `findings.md`
- `.workflow/dashboard.html`
- `.workflow/dispatch_suggestions.md`

generated outputs：
- `local_live_analysis/5-13-day-control-30min/stage5b_quote_anchor_diagnostic_0518T003/quote_anchor_diagnostic_summary.md`
- `local_live_analysis/5-13-day-control-30min/stage5b_quote_anchor_diagnostic_0518T003/bbo_source_drift.csv`
- `local_live_analysis/5-13-day-control-30min/stage5b_quote_anchor_diagnostic_0518T003/quote_distance_bucket_summary.csv`
- `local_live_analysis/5-13-day-control-30min/stage5b_quote_anchor_diagnostic_0518T003/post_only_reject_throttle_churn_summary.csv`
- `local_live_analysis/5-13-day-control-30min/stage5b_quote_anchor_diagnostic_0518T003/stale_latency_guard_summary.csv`
- `local_live_analysis/5-13-day-control-30min/stage5b_quote_anchor_diagnostic_0518T003/anchor_source_x_fill_markout.csv`
- `local_live_analysis/5-13-day-control-30min/stage5b_quote_anchor_diagnostic_0518T003/current_enforcement_gap_matrix.csv`
- `local_live_analysis/5-13-day-control-30min/stage5b_quote_anchor_diagnostic_0518T003/rounding_clamp_counterfactual.csv`
- `local_live_analysis/5-13-day-control-30min/stage5b_quote_anchor_diagnostic_0518T003/run_manifest.json`

action：
- 实现 `examples/binance_tick_mm/quote_anchor_diagnostic.py`，保持 read-only diagnostic 边界。
- 新增 focused test `examples/binance_tick_mm/test_quote_anchor_diagnostic.py`。
- 在 `5-13-day-control-30min` 上全量运行并生成 Step 5B quote-anchor / post-only 诊断产物。
- 没有修改 quote placement、fair/reservation、risk guards、live scripts、replay lifecycle 或 standard schema。
- 没有启动 live、没有补样本、没有运行策略实验。

主要发现：
- 样本规模：
  - decision rows：`47499`
  - submit label rows：`2516`
  - bookTicker anchor available rows：`47499`
  - top5 anchor available rows：`47499`
  - join stale decision rows：`432`
  - join missing / gap-crossed rows：`0 / 0`
- BBO source drift：
  - audit_depth vs bookTicker bid mismatch rate：`0.3524495252531632`
  - audit_depth vs bookTicker ask mismatch rate：`0.3531653297964168`
  - audit_depth vs bookTicker p99 abs drift：bid `142` ticks，ask `150` ticks
  - bookTicker vs top5_depth mismatch rate much smaller：bid `0.002652687424998421`，ask `0.006589612412892903`
  - 解释：sidecar 内 bookTicker/top5 depth BBO 彼此接近，但 live audit depth view 与 sidecar/as-of anchor view 存在大量逐行不一致；这支持继续保留 source-drift diagnostic，不支持直接把现有路径称为已实现 fast bookTicker hard anchor。
- rounding / clamp counterfactual：
  - current path vs audit_depth post-round risk rows：`0 / 47499`
  - T002 design path vs audit_depth post-round risk rows：`0 / 47499`
  - current path vs bookTicker post-round risk rows：`3675 / 47499`，rate `0.07737004989578727`
  - T002 design path vs bookTicker post-round risk rows：`0 / 47499`
  - current path vs top5_depth post-round risk rows：`3615 / 47499`，rate `0.07610686540769279`
  - T002 design path vs top5_depth post-round risk rows：`0 / 47499`
  - 解释：当前 target 对当前 audit_depth anchor 是干净的；但如果 hard anchor 切到 bookTicker/top5，必须实施 T002 的 side-conservative rounding + anchor clamp + re-check，否则会产生非零 post-only/crossed risk。
- current enforcement gap matrix：
  - 已满足：`top5_not_final_hard_anchor`
  - design gap：`fast_bbo_bookticker_hard_anchor`、`depth_bbo_guarded_fallback_only`、`side_conservative_rounding_and_post_round_recheck`
  - partial：`stale_latency_join_age_submit_suppression`、`reject_throttle_drop_cooldown_path`
- reject / throttle / churn：
  - decision reject reasons：`latency_guard=16528`，`quote_throttle=5996`，`api_interval_guard=1190`，`none=23785`
  - stage5 submit post_only_risk：`0 / 2516`
  - fast_cancel_churn：`1955 / 2516`
- quote placement / execution labels：
  - touch submit rows：`514`，step_back_gt1 submit rows：`1985`，step_back_1 submit rows：`17`
  - touch fill rate observed-only about `0.0647` for non-stale live_depth rows; step_back_gt1 fill rate about `0.0097`
  - touch observed markout is negative in this sample; this remains observed-only and not counterfactual fill proof.

当前判断：
- `0518T003` 诊断结果是 `diagnostic_only_not_ready_for_direct_implementation`。
- 它支持 T002 的设计方向：如果后续要把 fast BBO/bookTicker 作为 hard anchor，需要新增明确的 anchor arbitration、side-conservative rounding、post-clamp re-check、join-age/stale guard 和 reject handling。
- 它不支持直接创建 production quote-control implementation，也不支持 live promotion。
- 更合适的后续是 QA 后由总控决定是否开一个 default-off implementation design/implementation task，且 scope 应该很窄：先实现 anchor/clamp/re-check 的 default-off 或 diagnostic-first 版本。

verify：
- `python -m pytest examples/binance_tick_mm/test_quote_anchor_diagnostic.py`
- `python examples/binance_tick_mm/quote_anchor_diagnostic.py --help`
- `python examples/binance_tick_mm/quote_anchor_diagnostic.py --run-dir local_live_analysis/5-13-day-control-30min --output-dir local_live_analysis/5-13-day-control-30min/stage5b_quote_anchor_diagnostic_0518T003`
- `python3 .workflow/build_dashboard.py`
- `git diff --check`

done：
- 已实现 Step 5B read-only quote-anchor / post-only diagnostic runner。
- 已生成 T003 要求的全部产物。
- 已说明 quote-anchor / post-only diagnostic 的主要发现。
- 已说明 5A 设计假设中哪些被支持、哪些需要修改。
- 已明确当前 5A 五条约束中哪些已由现有代码/参数保证，哪些只是设计缺口。
- 已明确当前 rounding/clamp 路径与 T002 设计路径的只读差异。
- 已说明当前不足以直接创建 production/default-on implementation；若继续，只能开窄 scope default-off / diagnostic-first implementation task。
- 已明确未改策略、未启动 live、未补样本。

blockers：
- 无

commit：
- 8b701fb

提交信息：
- feat(binance-mm): add quote anchor diagnostics
```
