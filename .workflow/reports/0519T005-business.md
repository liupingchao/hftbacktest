```md
执行线程：
- 测试线程

任务ID：
- 0519T005

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0519T005.md`
- `.workflow/reports/0519T005-business.md`
- `task_plan.md`
- `progress.md`
- `findings.md`
- `.workflow/dashboard.html`
- `.workflow/dispatch_suggestions.md`
- `local_live_analysis/5-13-day-control-30min/stage8b_quote_update_diagnostic_0519T005/`

action：
- 将 `0519T005` 状态切到执行中并完成 Step 8B read-only diagnostic / implementation-planning。
- 读取 `0519T004` Step 8 design contract、Step 5C quote-anchor safety 结果、Step 6 closure / final calibration 和 Step 7 design boundary。
- 基于现有 `5-13-day-control-30min` artifacts 做只读聚合诊断，没有修改策略、replay、live、connector、schema 或默认配置。
- 生成 Stage 8B 诊断目录：
  - `quote_update_diagnostic_summary.md`
  - `implementation_planning_decision.json`
  - `decision_action_gate_summary.csv`
  - `planned_vs_actual_gate_matrix.csv`
  - `submit_churn_fill_regime_summary.csv`
  - `stale_bad_price_anchor_summary.csv`
  - `api_request_spacing_summary.csv`
  - `api_burst_summary.csv`
  - `min_move_quote_age_proxy_summary.csv`
  - `inventory_api_conflict_proxy_summary.csv`
  - `implementation_field_readiness.csv`
  - `run_manifest.json`

diagnostic result：
- conclusion: `default_off_helper_candidate`
- Step 9 status: `blocked_until_default_off_helper_or_instrumentation_boundary_is_accepted`
- 解释：
  - 当前样本显示足够的 quote update pressure 和 suppression pressure，支持后续开一个 default-off helper / instrumentation task。
  - 但缺少专用 `quote_update_*`、`token_bucket_state`、`cancel_readd_bucket`、`quote_age_ms`、post-only pre/post-check 和 `inventory_request_id` 等字段，不支持直接进入 full default-off quote-adjustment replay candidate。

key counts：
- decision rows: `47499`
- planned submit decision rows: `8971`
- actual submit decision rows: `2256`
- planned/action mismatch rows: `7186`
- latency guard rows: `16528`
- quote throttle rows: `5996`
- api interval guard rows: `1190`
- submit orders with labels: `2516`
- fast-cancel churn rows: `1955` / `2516` (`0.777027`)
- Stage 5C bid/ask clamped rows: `1394` / `2281`
- Stage 5C stale anchor rows: `65`
- Stage 5C post-only risk after re-check rows: `0`

observable fields：
- `action` / `planned_action` can proxy actual vs suppressed quote activity.
- `reject_reason` / `throttle_reason` expose latency, quote-throttle and API interval gates.
- Stage 5C diagnostic rows expose fast-anchor availability, guarded fallback, anchor age, stale anchor, clamp, suppress and post-clamp re-check counters.
- Stage 5 execution labels expose submit-level placement, latency, inventory score, recent reject/throttle counts, fast-cancel churn, fill horizons and fill-after-cancel labels.
- Stage 6 final calibration remains aligned enough for roadmap progression, but not for live promotion.

missing or proxy-only fields：
- Missing: `quote_update_intent`, unified `quote_update_reason`, `token_bucket_state`, `inventory_request_id`.
- Proxy-only: `min_move_passed`, `quote_age_ms`, `cancel_readd_bucket`, `latency_bucket`.
- Diagnostic-only: `anchor_age_ms`, post-clamp `post_only_post_check` from Stage 5C.

next-step recommendation：
- 不建议直接进入 Step 9。
- 建议如果继续 Step 8，先新建一个 default-off helper / instrumentation implementation task：
  - centralize quote-update intent/action/reason
  - record throttle/token/cancel-readd/post-only/inventory-request fields
  - preserve existing throttle/API/latency suppression semantics
  - keep default behavior unchanged
- Step 9 需要等待这个 helper / instrumentation boundary 或总控明确接受 `needs_more_instrumentation` 以外的替代边界。

verify：
- 诊断 artifacts 已生成并人工抽查 summary / decision JSON / gate summary / field readiness。
- 文档一致性检查：`task_plan.md`、`progress.md`、`findings.md` 已同步 Step 8B 结论和 Step 9 边界。
- `python3 .workflow/build_dashboard.py`
- `git diff --check`

done：
- Step 8B read-only diagnostic / implementation-planning 已完成。
- 结论四选一结果为 `default_off_helper_candidate`。
- 已明确 current churn/API/stale/bad-price regimes 哪些可观测、哪些字段缺失。
- 未授权策略实现、new replay sweep、live、default-on、Step 5C promotion 或 Step 9 promotion。

blockers：
- 无执行 blocker。
- 后续 Step 9 仍被 helper / instrumentation boundary 阻塞。

commit：
- 7c0c189

提交信息：
- docs(workflow): complete 0519T005 step 8b diagnostic
```
