```md
执行线程：
- 业务线程-python

任务ID：
- 0515T001

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0515T001.md`
- `.workflow/reports/0515T001-business.md`
- `examples/binance_tick_mm/replay_lifecycle_mismatch_diagnosis.py`
- `examples/binance_tick_mm/test_replay_lifecycle_mismatch_diagnosis.py`
- `task_plan.md`
- `progress.md`
- `findings.md`
- `.workflow/dashboard.html`
- `.workflow/dispatch_suggestions.md`
- `local_live_analysis/5-13-day-control-30min/stage6c_replay_lifecycle_mismatch_0515T001/**`

action：
- 新增 `examples/binance_tick_mm/replay_lifecycle_mismatch_diagnosis.py`：
  - 复用 Stage 6B 的 matched submit opportunity comparison unit。
  - 读取 live / replay execution rows，并生成：
    - matched submit state diff
    - replay-only fill cases
    - live-cancel / replay-fill cases
    - cancel-request / cancel-ack / fill / terminal timeline diff
    - terminal-state transition diff
    - cancel-ack delay diff
    - placement / inventory / latency grouped attribution
    - replay-only fill by horizon
    - cancel-race gap by bucket
- 新增 `examples/binance_tick_mm/test_replay_lifecycle_mismatch_diagnosis.py`：
  - 覆盖 required artifacts 输出
  - 覆盖 replay-only fill / live-cancel-replay-fill cases
  - 覆盖 grouped attribution outputs
- 在 `5-13-day-control-30min` 上全量运行 diagnosis runner，输出到：
  - `local_live_analysis/5-13-day-control-30min/stage6c_replay_lifecycle_mismatch_0515T001/`

run result：
- dataset：`5-13-day-control-30min`
- matched submit rows：`2516`
- replay-only fill rows：`120`
- live-cancel / replay-filled rows：`120`
- cancel-fill timeline diff rows：`135`
- terminal-state diff rows：`230`

main diagnosis：
1. replay-only fills are overwhelmingly live-canceled / replay-filled
   - `replay_only_fill_rows = 120`
   - `live_cancel_replay_fill_rows = 120`
   - 这说明当前 replay-only fills 并不是广义“live no-fill / replay fill”随机漂移，而是高度集中在 live 已经 canceled 的 submit opportunities 上。

2. replay-only fills are not mainly ultra-short-horizon events
   - replay-only fill by horizon：
     - `100ms`: `0`
     - `500ms`: `4`
     - `1000ms`: `6`
     - `5000ms`: `31`
   - 这支持“long-horizon persistence too optimistic”假设，而不是单纯短时 fill crossing / first-tick issue。

3. cancel timeline mismatch looks more like delayed terminal / cancel-ack semantics than submit matching error
   - 典型 live-canceled / replay-filled case：
     - live `cancel_ack_ts` 已经出现
     - replay 将同一 submit key 继续保留为 fill-eligible，并在更晚时刻给出 fill
   - 在 `cancel_fill_timeline_diff.csv` 中，多数 severe rows 都是：
     - live final state = `canceled`
     - replay final state = `filled`
     - replay cancel-to-fill delay 从数秒到数十秒，甚至更久
   - 这更像 replay terminal cutoff / cancel-ack / fill-eligibility window 问题，而不是 sample mismatch。

4. terminal-state mismatch is replay-side dominant
   - `terminal_state_transition_diff.csv` 中的主要模式是：
     - `canceled -> filled`
     - `canceled -> open_or_missing`
   - 很少看到反向的大量 `filled -> canceled`
   - 这说明 replay 并不是双向噪声，而是偏向“比 live 更难退出、更容易继续成交”。

5. mismatch is concentrated in certain strata, especially deeper placements
   - placement：
     - `step_back_gt1`:
       - replay_only_fill_rate `0.0569`
       - terminal_state_diff_rate `0.1113`
       - fill_after_cancel_gap `0.0564`
       - time_to_fill_gap_mean_ms `28198.95`
     - `touch`:
       - replay_only_fill_rate `0.0136`
       - fill_after_cancel_gap `0.0097`
   - latency buckets：
     - `q4` / `q5` have fill-after-cancel gaps around `0.0515-0.0518`
     - `q5` time_to_fill_gap_mean_ms `28059.16`
     - `q2` also severe at `25917.52`
   - 结论：
     - mismatch 不是完全均匀分布，更偏向 deeper step-back placement 和较慢 / 较老的 latency regimes。

6. markout observability mismatch is likely lifecycle-induced side effect
   - `0514T007` 已证明 fill_markout observable rows 差异大，但 fill_probability observable coverage 一致。
   - `0515T001` 进一步说明，很多新增 replay-observable fills 本身就是 replay-only fill rows。
   - 因此 markout observability mismatch 更像“replay 先多生成了 fills”，而不是独立的 future-price sampling bug。

what looks most like replay repair problems：
- cancel request / cancel ack / terminal state cutoff semantics
- replay fill eligibility persistence after cancel request
- long-horizon resting-order persistence bias

what is still not fully resolved：
- 仅从只读诊断表还不能区分：
  - 是 cancel-ack 应用太晚
  - 还是 fill generation 在 cancel-requested state 里本就过于激进
  - 或两者共同作用
- 还不能直接决定 repair 应该改哪个具体函数 / state machine 入口；这需要下一步单独 repair-task 继续定位。

next-step recommendation：
- 下一步不应补样本，也不应直接改策略。
- 应新建一个 replay lifecycle repair design / implementation task，最小目标是：
  - 收窄 cancel-requested order 的 fill eligibility
  - 对齐 terminal state / cancel-ack application semantics
  - 再用 `5-13-day-control-30min` 回归验证
- 更多 current-format sample 扩展只能放在 repair 之后，用于验证修正是否跨样本稳定。

verify：
- `python -m pytest examples/binance_tick_mm/test_replay_lifecycle_mismatch_diagnosis.py` -> `3 passed`
- `python examples/binance_tick_mm/replay_lifecycle_mismatch_diagnosis.py --help` -> passed
- `python examples/binance_tick_mm/replay_lifecycle_mismatch_diagnosis.py --run-dir local_live_analysis/5-13-day-control-30min --output-dir local_live_analysis/5-13-day-control-30min/stage6c_replay_lifecycle_mismatch_0515T001` -> passed
- 人工检查输出目录包含 required artifacts -> passed
- `python3 .workflow/build_dashboard.py` -> passed
- `git diff --check` -> passed

done：
- `0515T001` 已实现 read-only mismatch diagnosis runner 和 focused tests。
- 已在 `5-13-day-control-30min` 上完成 full-run diagnosis。
- 主结论是：
  - replay-only fills 高度集中在 live-canceled / replay-filled cases
  - mismatch 更像 cancel/terminal lifecycle persistence 问题，不像 sample coverage 问题
  - deeper step-back placement 与部分 latency bucket 是主要 hot spots
  - markout observability mismatch 更像 lifecycle-induced side effect
- 本任务未修 replay、未新跑 replay sweep、未补样本、未改策略、未启动 live。

blockers：
- 无执行阻塞。
- 诊断后的实现阻塞仍在：真正的 replay repair 需要独立任务。

commit：
- 待提交

提交信息：
- 待提交
```
