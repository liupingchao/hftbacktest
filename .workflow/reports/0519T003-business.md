```md
执行线程：
- 业务线程-docs

任务ID：
- 0519T003

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0519T003.md`
- `.workflow/reports/0519T003-business.md`
- `task_plan.md`
- `progress.md`
- `findings.md`
- `.workflow/dashboard.html`
- `.workflow/dispatch_suggestions.md`

action：
- 新建并派发 `0519T003`。
- 执行 Step 7 planning-only inventory / execution model redesign contract。
- 读取 `0519T002` QA 后的 Step 6 closure boundary。
- 将 Step 7 design contract 写入 `task_plan.md`、`progress.md`、`findings.md`。
- 没有改策略代码、没有跑新实验、没有启动 live、没有修改 replay/schema/connector。

design decision：
- Step 7 是 design-only 完成，不是 implementation。
- 目标是把 one-way maker 的 inventory behavior 从隐式对称报价，改为明确的 inventory band / recovery regime / cycle diagnostics 合同。
- 正常目标是接近 flat / one-order-quantity band。
- 超过 one order quantity 后进入 recovery regime：
  - 优先通过 reservation/fair shift、spread widening、size reduction、recovery-side preference、add-side suppression 设计候选
  - 不直接授权 aggressive exit、TTL 或 triple-barrier 实现
- Zero-crossing 应作为 inventory-cycle reset 语义，用于后续诊断 cycle duration、max excursion、recovery fills、recovery markout 和 clean zero-crossing。
- AS-style dynamic spread / dynamic order amount 只能作为后续 default-off candidate，输入必须是 decision-time-visible volatility、fill intensity、inventory、latency、lifecycle proxy。
- TTL / triple-barrier 只保留为后续 default-off design candidate，需要 pricing/fill/lifecycle evidence 后才能进入实现任务。

required future inputs and audit fields：
- Decision-time-visible inputs only:
  - current position / notional
  - target and working quote ticks
  - fair/reservation signals visible at decision time
  - volatility / spread / top-of-book or top5 size-age proxies
  - latency / stale / join-age flags
  - in-flight / cancel-requested / recent fill-cancel lifecycle state
- Required audit / diagnostic fields before implementation:
  - inventory band
  - inventory cycle id
  - skew regime
  - quote-side suppression reason
  - size multiplier
  - spread multiplier
  - reservation shift
  - TTL / barrier state
  - recovery-mode marker

boundaries：
- No strategy implementation.
- No default-on behavior.
- No live run or live collection.
- No replay sweep or new experiment.
- No exact queue proof claim.
- No use of future markout, audit replay overlay, or `4948`-specific logic as live decision inputs.
- Step 6 remains only event-classification / lifecycle-proxy closed; not promotion or live readiness.

next-step recommendation：
- `0519T003` QA 通过后，建议进入 Step 8 design-only quote-update / API-limit hygiene。
- 不建议立刻实现 Step 7 controls，因为 Step 8 需要先约束 quote-update mechanics、replace/cancel behavior、API/churn hygiene 和 stale/bad-price handling。
- Step 9 仍需等待 Step 7 / Step 8 design boundaries 都通过后，才可作为 default-off offline replay experiment。

verify：
- 文档一致性检查：`task_plan.md`、`progress.md`、`findings.md` 对 Step 7 design contract 不冲突。
- `python3 .workflow/build_dashboard.py`
- `git diff --check`

done：
- Step 7 planning/design 已完成并等待 QA。
- 下一步建议明确为 Step 8 design-only。
- 不允许事项已明确：no code, no experiment, no live, no default-on, no promotion。

blockers：
- 无执行 blocker。
- 后续 implementation / experiment 仍受 Step 8 design、Step 9 default-off offline replay 和更多 current-format samples 限制。

commit：
- 46d114c

提交信息：
- docs(workflow): add step 7 inventory design contract
```
