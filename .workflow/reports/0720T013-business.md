# 业务线程执行回报

执行线程：
- 总控 auto-loop / 业务执行线程

任务ID：
- 0720T013

状态：
- 待验收

更新时间：
- 2026-07-20 09:00 CST

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0720T013.md`
- `examples/hyperliquid/cross_exchange_t024_same_window_acceptance.py`
- `examples/hyperliquid/hyperliquid_tiny_live_m2_public_watcher.py`
- `examples/hyperliquid/test_cross_exchange_t024_same_window_acceptance.py`

action：
- Producer 的 immediate-guard CSV 新增直接 `event_sequence`，并停止为未执行 guard 的窗口写无身份占位行。
- Acceptance 对每个 trigger、anti-drift、immediate-guard、edge 和 attempt 行严格解析 canonical positive event/attempt identity：
  - malformed、missing、duplicate、orphan identity 不再跳过校验；
  - trigger event 必须唯一，非 trigger 行只能连接 `trigger_found=true` 的 event；
  - attempt/attempt_id 必须同时有效且一致；
  - attempt key 必须符合 exact task、`window_01` 和 attempt suffix。
- 建立双向 causal joins：
  - anti-drift block 与 trigger status/reason/phase 精确一致；
  - immediate-guard pass/fail 与 trigger stage 精确一致；
  - edge pass/block 与 trigger status/reason 精确一致；
  - attempt 的 guard、edge、skip reason 与 trigger 因果链精确一致；
  - pre/post anti-drift phase 决定 guard 行必须为 0 或 1，额外行 fail closed。
- 每个 submitted attempt 必须连接 `live_window_called=true` 的授权 trigger；零授权与两个 submissions 同时存在时 fail closed。
- Two-sided manager 保留两个 side-specific canonical attempt identities，并允许二者连接同一个授权 event。
- Legacy guard timestamp bridge 默认关闭，只能通过显式 CLI 开关且 exact task/source 同时为：
  - task `0719T011`
  - source `d8e22c2d9288fef86707d9b26f7791d7d8711c09`
- 新增 synchronized/resealed 对抗测试：
  - malformed event/attempt；
  - legacy schema downgrade；
  - noncanonical attempt key；
  - zero authorization plus two submissions；
  - anti-drift、guard、edge cross-matrix reason/status drift。

verify：
- Exact implementation commit：
  - `5b482929a09c4402b5682fc9c7c130aafbb25977`
- Focused acceptance/watcher/orchestrator/manager：
  - `243 passed in 16.76s`
- Full Hyperliquid：
  - `706 passed in 35.44s`
- Modified modules/tests `py_compile`：pass。
- Acceptance CLI `--help`：pass，并显示 explicit legacy bridge 开关。
- Watcher CLI `--help`：pass。
- Implementation commit `git show --check`：pass。
- T011 byte-exact offline replay：
  - legacy guard bridge authorized only for exact T011 task/source；
  - provenance `112 pass / 0 fail`；
  - config `70 pass / 1 fail`；
  - decision `16 pass / 27 fail`；
  - lifecycle `34 pass / 27 fail`；
  - independent summary validation reasons `[]`；
  - candidate evaluations `2564`，trigger rows `26`；
  - anti-drift `42 pass / 5 block`；
  - immediate guard `10 pass / 11 fail`；
  - edge gate `0 pass / 10 block`；
  - candidate rows / manager identity / submissions `21 / 1 / 0`；
  - private read/order/cancel `true / false / false`；
  - final recommendation 继续 blocked。
- 未执行 live、private、account、order、cancel、network、remote 或 service。

done：
- T012 QA 的 malformed identity、authorization gap 和 cross-matrix causal drift 均已转为 fail-closed。
- 新 producer evidence 直接携带 guard event identity；legacy compatibility 不能被普通新 artifact 自行降级启用。
- T011 旧证据可确定性重放且没有新增 path/source/checksum failure，但仍因零 submission 和缺失 lifecycle 正确 blocked。
- Strategy formula、threshold、risk cap、activation 和 multi-level 状态未改变。

blockers：
- 当前唯一流程节点是独立 QA 验收 T013。
- T013 QA 通过前不得启动新的 bounded live window。
- Principal Task 7/12 lifecycle 和 multi-level unlock 仍未完成。

commit：
- `5b482929a09c4402b5682fc9c7c130aafbb25977`

提交信息：
- `Enforce strict decision evidence joins`
