执行线程：
- 业务线程

任务ID：
- 0706T004

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `docs/cross_exchange_maker_mvp_plan.md`
- `docs/cross_exchange_mvp_auto_loop_plan.md`
- `task_plan.md`
- `progress.md`
- `findings.md`
- `.workflow/tasks/0706T004.md`
- `.workflow/reports/0706T004-business.md`

action：
- 对照 `0706T003 / 0625T009` 最新 QA 结果，检查 MVP 目标和 auto-loop plan。
- 更新 `docs/cross_exchange_maker_mvp_plan.md`，明确当前 T009 仅支持一单 submit/resting/cancel/open-orders 窄事实，完整 fill/cost/PnL replay 仍需要新 live evidence 和显式授权。
- 更新 `docs/cross_exchange_mvp_auto_loop_plan.md`，把当前 workflow 起点改为 `0706T004 已通过`，并保留底层执行校准事实 `0706T003 已通过`，同时把下一步路线拆成：
  - `0625T010-SCOPED` supported-fact same-window replay acceptance，可自动创建执行。
  - 完整 `0625T010`，必须等待新的完整 live evidence 和显式授权。
- 更新 `task_plan.md`、`progress.md`、`findings.md`，记录 `0706T004` 路线图刷新结论和下一步任务顺序。

verify：
- 人工复核路线图事实一致性：
  - T003-T007 为已验收 upstream lineage。
  - T008/T009 当前仅支持一单 submit/resting/cancel/open-orders。
  - T010-SCOPED 不能解锁 T011。
  - full T010/T011/T012 仍需新 live evidence 和显式授权。
- `git diff --check` 通过。

done：
- MVP 目标没有降低：完整目标仍包含 tiny-live lifecycle/economics evidence、same-window replay 和 replay/live acceptance。
- auto-loop 下一步更新为 `0706T005 / 0625T010-SCOPED Supported-Fact Same-Window Replay Acceptance`。
- 任意新增 live-submit、repeated-window、fill-seeking、closer-to-market placement、size change 或 quote-envelope change 均明确要求新正式任务和显式授权。

blockers：
- 无

commit：
- 无

提交信息：
- 无
