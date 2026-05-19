```md
执行线程：
- 业务线程-docs

任务ID：
- 0519T004

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0519T004.md`
- `.workflow/reports/0519T004-business.md`
- `task_plan.md`
- `progress.md`
- `findings.md`
- `.workflow/dashboard.html`
- `.workflow/dispatch_suggestions.md`

action：
- 将 `0519T004` 状态切到执行中并完成 planning-only Step 8 design contract。
- 读取 `0519T003` QA、Step 5C quote-anchor safety 结论和 Step 8 计划边界。
- 将 quote-update mechanics / API-limit hygiene 设计合同写入 `task_plan.md`、`progress.md`、`findings.md`。
- 没有改策略代码、没有跑新实验、没有启动 live、没有修改 replay/schema/connector。

design decision：
- Step 8 是 design-only 完成，不是 implementation。
- quote-update mechanics should be driven by observable safety / usefulness triggers:
  - bad-price ticks
  - minimum quote move
  - quote age
  - stale or missing anchor
  - join-age / latency regime
  - post-clamp post-only risk
  - inventory regime request from Step 7
- Preferred future action ordering:
  - hold quote when price is still useful and API/churn budget should be preserved
  - modify/replace in place if supported and safer than cancel+new
  - cancel+new only when quote is materially unsafe, stale, crossed-risky, inventory-worsening, or past bounded age
- GTX/post-only reject remains an exchange backstop and diagnostic bucket, not normal control flow.
- Step 5C quote-anchor safety remains default-off / diagnostic-first unless a later implementation task explicitly changes that boundary.
- Step 7 inventory controls may request quote changes only through shared update-intent fields. They must not bypass anti-churn, throttle, token bucket, stale-anchor suppression, or post-only re-check.

required future controls and fields：
- Anti-churn controls:
  - per-side min tick move
  - min quote age
  - max cancel/re-add rate
  - in-flight order guard
  - cancel-pending guard
  - recent reject/throttle cooldown
  - emergency stale/bad-price override
- API hygiene controls:
  - token bucket
  - request spacing
  - per-action rate budgets
  - cancellation-limit risk
  - reject/throttle/drop buckets
  - observable degraded modes
- Required audit / diagnostic fields:
  - quote_update_intent
  - quote_update_action
  - quote_update_reason
  - min_move_passed
  - quote_age_ms
  - anchor_age_ms
  - join_age_ms
  - latency_bucket
  - throttle_state
  - token_bucket_state
  - cancel_readd_bucket
  - reject/throttle/drop cause
  - post_only_pre_check
  - post_only_post_check
  - inventory_request_id

boundaries：
- No strategy implementation.
- No default-on behavior.
- No live run or live collection.
- No replay sweep or new experiment.
- No Step 5C default-on promotion.
- No reliance on GTX/post-only reject as normal control flow.
- No use of Step 6/7 single-sample evidence as live readiness or quote-adjustment promotion.

next-step recommendation：
- `0519T004` QA 通过后，不建议直接进入 Step 9。
- 建议新建一个 narrow Step 8B read-only diagnostic / implementation-planning task over existing artifacts。
- Step 8B 应量化 current churn/API/stale/bad-price regimes，并决定后续应是 no-change、default-off helper implementation，还是 full default-off replay candidate。
- Step 9 仍需等待 Step 8B/implementation boundary 和后续 QA 后，才可作为 default-off offline replay experiment。

verify：
- 文档一致性检查：`task_plan.md`、`progress.md`、`findings.md` 对 Step 8 design contract 不冲突。
- `python3 .workflow/build_dashboard.py`
- `git diff --check`

done：
- Step 8 planning/design 已完成并等待 QA。
- 下一步建议明确为 Step 8B read-only diagnostic / implementation-planning。
- 不允许事项已明确：no code, no experiment, no live, no default-on, no promotion。

blockers：
- 无执行 blocker。
- 后续 implementation / experiment 仍受 Step 8B diagnostics、Step 9 default-off offline replay 和更多 current-format samples 限制。

commit：
- de68bb8

提交信息：
- docs(workflow): complete step 8 quote update design
```
