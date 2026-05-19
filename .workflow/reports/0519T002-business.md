```md
执行线程：
- 业务线程-docs

任务ID：
- 0519T002

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0519T002.md`
- `.workflow/reports/0519T002-business.md`
- `task_plan.md`
- `progress.md`
- `findings.md`
- `.workflow/dashboard.html`
- `.workflow/dispatch_suggestions.md`

action：
- 将 `0519T002` 状态切到执行中并完成 planning-only closure decision。
- 读取 `0519T001` QA、业务回报和 final calibration artifacts。
- 没有运行新实验、没有改代码、没有启动 live。
- 将 Step 6 closure boundary 写入 `task_plan.md`、`progress.md`、`findings.md`。

closure decision：
- Step 6 final state: `closed_for_roadmap_progression_requires_more_samples_for_promotion`。
- 原 Stage 6B 的 broad blocker 已解决到可继续路线图推进：
  - `diagnostic_only_gap_too_large` 已改善为 `requires_more_current_format_samples`
  - matched submit coverage `2516/2516`
  - matched price tick equality `2516/2516`
  - matched qty equality `2516/2516`
  - live/replay filled `53/54`
  - live/replay fill-after-cancel `16/15`
- Step 6 只关闭 event-classification / lifecycle-proxy 层面，不关闭以下事项：
  - live readiness
  - quote-adjustment promotion
  - exact queue proof
  - generalized queue/touch repair
  - single-sample PnL 结论

remaining boundaries：
- `4948` / `28940|sell` 继续作为 design-only residual 搁置。
- queue/priority、opportunity cost、realized PnL decomposition 仍是 observed-only proxy。
- time-to-fill magnitude 和 cancel-to-fill delay magnitude 仍存在非零 gap。
- 更多 current-format samples 仍然需要，但它们不阻塞 Step 7 / Step 8 design work。

next-step recommendation：
- `0519T002` QA 通过后，建议先启动 Step 7：design-only inventory / execution model redesign。
- Step 8 可作为后续 design-only quote-update / API-limit hygiene 任务。
- Step 9 只能在 Step 7 / Step 8 design boundaries 被接受后启动，并且只能是 default-off offline replay experiment。
- 不建议当前立即补样本作为唯一下一步；补样本应在 Step 9 promotion-style decision、live micro-test decision 或 queue/touch repair evidence 需要时再作为前置。

verify：
- 文档一致性检查：`task_plan.md`、`progress.md`、`findings.md` 对 Step 6 closure 结论一致。
- `python3 .workflow/build_dashboard.py`
- `git diff --check`

done：
- Step 6 的最终状态已经明确。
- Step 7 / Step 8 / Step 9 的进入边界已经明确。
- residual queue/touch issue 的处理边界已经明确。
- 本任务是 planning-only，没有代码、实验、live 或 repair。

blockers：
- 无执行 blocker。
- 证据限制仍在：promotion / live readiness / generalized queue repair 前仍需要更多 current-format samples。

commit：
- 4189e3d

提交信息：
- docs(workflow): close step 6 boundary decision
```
