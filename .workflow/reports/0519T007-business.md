```md
执行线程：
- 业务线程-python

任务ID：
- 0519T007

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0519T007.md`
- `.workflow/reports/0519T007-business.md`
- `task_plan.md`
- `progress.md`
- `findings.md`
- `.workflow/dashboard.html`
- `.workflow/dispatch_suggestions.md`

action：
- 将 `0519T007` 状态切到执行中并完成 Step 9A default-off quote-adjustment replay experiment design contract。
- 复核 `0519T006` QA、`0519T005` Step 8B、`0519T004` Step 8 design、`0519T003` Step 7 design、`0519T002` Step 6 closure 和 `0518T004` Step 5C quote-anchor safety 边界。
- 制定 Step 9A 设计合同：candidate matrix、decision-time-visible inputs、metrics、artifacts、Step 9B implementation boundary、acceptance gate、non-goals。
- 未实现 runner、未改策略代码、未跑 replay、未启动 live、未开启 default-on、未做 promotion。

design contract：
- Candidate matrix:
  - `baseline_control`: no-change replay validation; proves runner and metrics reproduce baseline before any candidate interpretation.
  - `fair_reservation_shift`: use decision-time-visible fair/reservation/pricing signals to move both sides or one side conservatively; expected to reduce adverse markout, risk is directional bias and lower fill rate.
  - `inventory_reservation_shift`: use position, inventory band, recovery regime and future `inventory_request_id` placeholder to request reservation shift or recovery-side preference; expected to reduce max inventory excursion, risk is missed spread capture.
  - `spread_widening`: widen on stale, latency, adverse, inventory-worsening, or post-only-risk regimes; expected to reduce toxic fills and post-only risk, risk is lower fill probability.
  - `size_reduction_or_add_side_suppression`: reduce size or suppress add-side in high inventory/API/churn pressure regimes; expected to reduce inventory worsening and churn, risk is lower participation.
  - `stale_latency_no_fresh_add`: block fresh add-side quote when latency/join/anchor age is unsafe while preserving cancel/reduce-side safety behavior; expected to reduce bad-price exposure, risk is too much inactivity.
  - `min_move_quote_age_churn_guard`: hold quotes when tick move is small, quote age is low, token bucket pressure is high, or recent cancel-readd churn is high; expected to reduce API/churn pressure, risk is stale quote persistence.
  - `post_only_safety_interaction`: replay interaction with Step 5C side-conservative rounding, anchor clamp, stale/missing anchor suppression and post-clamp re-check; expected to keep post-only risk observable, risk is over-clamping / too passive quoting.
- Decision-time-visible inputs:
  - market view fields, target ticks, working order state, position/inventory score, latency/join/anchor age, throttle/API state, Step 5C safety diagnostics, T006 quote-update fields, and existing strategy risk config.
  - Disallowed as decision inputs: future markout, future fill outcome, audit replay overlay labels, exact queue position claims, `4948`-specific repair logic, or same-sample PnL feedback.
- T006 audit/explanation layer:
  - `quote_update_intent/action/reason` explain proposed vs executed quote updates.
  - `min_move_passed`, `quote_age_ms`, `join_age_ms`, `anchor_age_ms`, `latency_bucket` explain quote freshness and stale/latency regimes.
  - `throttle_state`, `token_bucket_state`, `cancel_readd_bucket`, `reject_throttle_drop_cause` explain API/churn/drop causes.
  - `post_only_pre_check`, `post_only_post_check` explain post-only safety.
  - `inventory_request_id` remains a passive placeholder until a separate inventory implementation task.

metrics and acceptance：
- PnL / fee / spread capture:
  - report gross PnL, net PnL after fees, realized spread capture, and fee drag.
  - single-sample PnL can only classify a candidate as diagnostic; it cannot authorize promotion.
- Fill quality:
  - fill probability and time-to-fill by side, quote distance, latency bucket, inventory band and candidate.
  - side-adjusted markout after fill at 100ms/500ms/1s/5s.
  - adverse-selection buckets and spread-capture distribution.
- Lifecycle / cancel-fill:
  - cancel-fill count, fill-after-cancel count, cancel-to-fill delay, cancel-race buckets and residual Step 6 mismatch notes.
  - Step 6 remains event-classification / lifecycle-proxy closure, not exact queue proof.
- Inventory:
  - inventory cycle id, zero-crossing, max excursion, time in band, recovery duration, recovery-side fills, markout while reducing inventory.
- API / churn:
  - API request count, token bucket pressure, throttle/reject/drop count, min-move fail/pass, quote age distribution, cancel-readd bucket, fast-cancel churn, planned/action mismatch.
- Safety:
  - stale/missing anchor, clamp/suppress counts, post-only pre/post-check, bad-price/crossed-risk counters, Step 5C safety interaction.
- Coverage:
  - action-path coverage, audit-field coverage, candidate-decision coverage, market-view gate status, lifecycle label coverage.
- Candidate classification:
  - `no_effect`
  - `worse_due_to_churn_or_fill_quality`
  - `promising_but_single_sample`
  - `blocked_by_replay_or_market_view`
  - `needs_more_instrumentation`

Step 9B minimum implementation boundary：
- Implement only a default-off offline runner over the accepted sample first.
- The runner may simulate candidate policies offline and/or invoke default-off replay configs, but must not change live defaults or production behavior.
- Required initial sample: `5-13-day-control-30min` for runner validation only.
- Required outputs:
  - `run_manifest.json`
  - `candidate_matrix.csv` / `candidate_matrix.json`
  - `candidate_summary.json`
  - `candidate_metrics.csv`
  - `fill_quality_by_candidate.csv`
  - `inventory_cycle_metrics.csv`
  - `api_churn_metrics.csv`
  - `post_only_safety_metrics.csv`
  - `action_path_coverage.csv`
  - `audit_field_coverage.csv`
  - `candidate_decision_samples.csv`
  - `acceptance_decision.md`
- Step 9B must pass maker acceptance / market-view gate where applicable and must report whether any replay/live lifecycle assumptions remain blocking.
- Step 9B must end in a diagnostic decision; no live or promotion decision is allowed.

Step 9B dispatch gate：
- `0519T007` QA must pass.
- T007 QA must confirm candidate matrix, metrics, artifacts, input restrictions and non-goals are covered.
- Step 9B task must explicitly remain default-off offline-only.
- Step 9B must not require new sample collection before runner validation unless QA finds a design/data blocker.

non-goals：
- no implementation
- no replay sweep
- no live
- no default-on
- no production strategy behavior change
- no Step 5C promotion
- no inventory-control implementation
- no queue/touch repair
- no current-format sample expansion
- no promotion or live-readiness claim

verify：
- 人工核对 `.workflow/reports/0519T006-qa.md`、`0519T005` Step 8B 结论、Step 7 / Step 8 / Step 5C / Step 6 边界均已纳入设计合同。
- `python3 .workflow/build_dashboard.py`
- `git diff --check`

done：
- Step 9A design-only 合同已完成。
- 已列出 candidate matrix、metrics、artifacts、Step 9B implementation boundary、non-goals。
- 建议 QA 通过后派发 Step 9B：最小范围是 default-off offline runner implementation over `5-13-day-control-30min` for runner validation。
- 明确 no implementation, no replay sweep, no live, no default-on, no promotion。

blockers：
- 无。

commit：
- b9bdb6b

提交信息：
- docs(workflow): complete 0519T007 step 9a design
```
