```md
执行线程：
- 业务线程-python

任务ID：
- 0515T002

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0515T002.md`
- `.workflow/reports/0515T002-business.md`
- `task_plan.md`
- `progress.md`
- `findings.md`
- `.workflow/dashboard.html`
- `.workflow/dispatch_suggestions.md`

action：
- 读取 `0515T001` business / QA、`0514T008` planning contract、Stage 6B calibration summary 和 `0515T001` diagnosis tables。
- 将 replay lifecycle mismatch 的诊断结果收敛成 repair design contract：
  - 排序 repair hypotheses
  - 定义最小 repair scope
  - 明确暂不修的事项
  - 写出验证门槛 / 成功标准 / 失败标准
  - 约束后续真正 implementation task 的输入、输出和 non-goals
- 保持 planning-only 边界：
  - 未修改 replay fill model / lifecycle state machine / cancel-ack handling
  - 未新跑 replay sweep
  - 未补采样本
  - 未改策略
  - 未启动 live

problem framing：
- `0515T001` 已经把 mismatch 从“样本不足”收敛到“replay lifecycle semantics mismatch”：
  - replay-only fill rows：`120`
  - live-cancel / replay-filled rows：`120`
  - terminal-state diff rows：`230`
  - replay-only fills 并不集中在 ultra-short horizon，而更偏向 longer persistence
- 因此 `0515T002` 的目标不是再确认问题是否存在，而是把“该修什么、先修什么、怎么验证”写成可执行合同。

repair hypothesis ranking：
1. P0: cancel-requested fill eligibility window too permissive
   - diagnosis evidence：
     - replay-only fills 几乎全部是 `live-canceled / replay-filled`
     - many rows show live cancel-ack already reached while replay still fills later
   - likely implication：
     - replay 在 cancel-requested state 下继续允许 fills 的窗口过长

2. P0: terminal-state cutoff semantics diverge from live
   - diagnosis evidence：
     - dominant transition `canceled -> filled`
     - secondary transition `canceled -> open_or_missing`
   - likely implication：
     - replay 的 cancel terminalization 和 live 不一致，导致订单更晚退出

3. P1: long-horizon resting-order persistence too optimistic
   - diagnosis evidence：
     - replay-only fill by horizon: `100ms=0`, `500ms=4`, `1000ms=6`, `5000ms=31`
     - `step_back_gt1` rate明显高于 `touch`
   - likely implication：
     - replay 对 deeper step-back orders 的长期存活/成交概率偏乐观

4. P1: cancel-ack application timing lags downstream lifecycle updates
   - diagnosis evidence：
     - cancel timeline diff rows `135`
     - large delay gaps after cancel request
   - likely implication：
     - cancel-ack 与 fill/terminal ordering 可能不同步，或消费顺序与 live 不一致

5. P2: remaining qty / partial lifecycle bookkeeping
   - diagnosis evidence：
     - 当前样本 `partial_fill=0`，所以它不是主要问题
   - implication：
     - 不应作为 first repair target，只保留为 secondary verification item

minimal repair scope recommendation：
- must-fix first：
  1. cancel-requested state 的 fill eligibility cutoff
  2. cancel-ack / canceled terminal-state semantics
  3. long-horizon persistence bias for step-back placements
- defer for later：
  1. exact queue / priority modeling
  2. full MBO-style queue position proof
  3. sample expansion before first repair regression
  4. quote-adjustment / strategy-side redesign

recommended implementation split：

1. first task: repair implementation (narrow)
   - scope:
     - replay-side cancel-requested fill eligibility
     - replay terminal-state cutoff / cancel-ack semantics
     - no strategy changes
     - no new sample collection
   - must not:
     - change quote logic
     - change fair/reservation
     - claim queue exactness

2. second task: repair regression validation
   - rerun the same `5-13-day-control-30min` acceptance / Stage 6B / Stage 6C chain
   - compare before/after:
     - replay_only_fill_rows
     - live_cancel_replay_fill_rows
     - final-state gaps
     - fill-after-cancel gap
     - time-to-fill gaps
     - placement / latency hot spots

3. third task: only if repair improves core gaps
   - add one or more extra current-format samples
   - use them to test whether repair generalizes

success criteria for later repair implementation task：
- core counters should improve materially on `5-13-day-control-30min`:
  - replay-only fill rows should drop materially from `120`
  - live-cancel / replay-filled rows should drop materially from `120`
  - `canceled -> filled` transition should shrink materially
  - fill-after-cancel-request rate gap should shrink materially from about `0.0465`
  - time-to-fill gap in step-back / slow-latency buckets should narrow materially
- no regression on:
  - matched submit coverage
  - short-horizon fill comparability that was already roughly aligned

failure criteria：
- repair only shifts rows from `filled` to `open_or_missing` without shrinking lifecycle mismatch
- repair improves aggregate counts but leaves `step_back_gt1` / `q4-q5 latency` hot spots largely unchanged
- repair reduces replay fills by broadly suppressing order lifecycle rather than aligning cancel/terminal semantics
- repair requires strategy-rule changes to look better

required validation artifacts for later implementation task：
- before/after comparison over the same sample:
  - `replay_only_fill_cases.csv`
  - `live_cancel_replay_fill_cases.csv`
  - `cancel_fill_timeline_diff.csv`
  - `terminal_state_transition_diff.csv`
  - `state_diff_by_placement.csv`
  - `state_diff_by_latency.csv`
  - `replay_only_fill_by_horizon.csv`
- summary documents:
  - repair regression summary markdown
  - explicit success/failure callout

repair touchpoint candidates：
- `examples/binance_tick_mm/backtest_tick_mm.py`
  - replay-side lifecycle event handling near cancel / fill / terminal transitions
- `examples/binance_tick_mm/strategy_core.py`
  - lifecycle tracker / cancel-request bookkeeping semantics
- note:
  - `0515T002` does not authorize touching them yet; it only identifies them as the most likely repair areas

non-goals for the later repair implementation task：
- no fair-price / reservation / quote placement change
- no live sample collection
- no Stage 6J sweep
- no queue exactness claim
- no quote-adjustment promotion

decision recommendation：
- next formal task should be a narrow replay lifecycle repair implementation task, not another diagnosis task and not a sample-collection task.
- if that repair task cannot be kept narrow, total controller should split it into:
  - repair implementation
  - repair regression validation

verify：
- 人工检查 `.workflow/tasks/0515T002.md` 和 `.workflow/reports/0515T002-business.md` 是否完整。
- `python3 .workflow/build_dashboard.py`
- `git diff --check`

done：
- 已形成 replay lifecycle repair design contract。
- 已明确 repair hypotheses、优先级、最小 repair scope、成功/失败标准、验证 artifacts 与后续 implementation 边界。
- 已明确本任务未实现 replay 修复、未运行新 replay、未补采样本、未改策略、未启动 live。

blockers：
- 无

commit：
- 待提交

提交信息：
- 待提交
```
