```md
执行线程：
- 业务线程-python

任务ID：
- 0511T001

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 正常验收。重点检查本任务是否只产出设计合同、是否禁止未来数据作为 live 决策输入、是否明确后续 0511T002 的实现边界和验收门禁。

files：
- .workflow/tasks/0511T001.md
- .workflow/reports/0510T001-business.md
- .workflow/reports/0510T002-business.md
- .workflow/reports/0510T001-qa.md
- .workflow/reports/0510T002-qa.md
- examples/binance_tick_mm/strategy_core.py
- examples/binance_tick_mm/live_tick_mm.py
- examples/binance_tick_mm/backtest_tick_mm.py
- docs/5-8-future-plan.md
- docs/stage6i-cancel-requested-fill-risk-plan.md
- findings.md
- progress.md

action：
- 读取 0511T001 任务合同，确认本任务是 design-only，不实现代码。
- 读取 0510T001/0510T002 business + QA 结果，确认前置证据链已完成：单样本闭环通过、跨样本 Stage 6J replay 完成、两者均不允许 live micro test。
- 读取 `strategy_core.py` 中 `cancel_race_guard_side_blocks` 和 `decide_actions`，确认现有 guard 形状是 side block -> add-side only block -> reduce-side allowed。
- 读取 live/backtest 调用点，确认两边已经共用 `strategy_core.py` 的 guard helper 和 `decide_actions`，但 signal-to-target 与 guard context assembly 仍在 loop 内重复。
- 读取 Stage 6J 文档，确认 adverse-selection timing rule 还没有被当前 replay 直接验证，T002 只支持继续做 default-off diagnostic design/implementation。

verify：
- 人工检查：本回报未修改 `strategy_core.py`、`live_tick_mm.py`、`backtest_tick_mm.py`、`audit_schema.py`、`stage6j_replay.py`。
- 人工检查：设计合同明确禁止 future mid、future markout、replay-only overlay 作为 live 决策输入。
- 人工检查：设计合同明确 live/backtest 必须复用同一个 `strategy_core.py` helper。
- 人工检查：设计合同明确第一版只允许 add-side block，reduce-side 必须允许。
- 人工检查：设计合同明确 0511T002 只能 default-off 实现与离线 replay，不允许 live micro test。

done：
- 当前分层判断：足以支撑 0511T002 的窄范围 default-off guard 接入。原因是 live/backtest 已经共享 `cancel_race_guard_side_blocks`、`decide_actions` 和 audit row builder；新增 adverse timing guard 可以沿用这个 shared helper + side block 的形状。
- 当前分层不足：signal-to-target 生成和 guard context assembly 仍分别存在于 live/backtest loop；0511T002 不应大重构，只能新增小 helper 和字段。后续如规则继续增多，应单独做 `SignalSnapshot`、`GuardState`、`DecisionContext`、`DecisionResult` 重构任务。
- 目标 source-path：cancel-requested fill、inventory-reducing cancel race、target deterioration、pending cancel toxic window。当前证据显示 `5-10-day-control-1h-06` 更偏 adverse-selection / inventory-reducing cancel race；T002 跨样本 replay 仍是 `diagnostic_only_no_promotion`。
- 允许输入：当前 decision timestamp 可见的 working orders、target bid/ask ticks、working order price tick、position、inflight exposure、cancel-requested side qty、last side cancel-fill timestamp、order age、feed/order latency signal、cancel latency bucket、short-horizon historical drift/rolling toxicity state。
- 禁止输入：future mid、future markout、未来 fill 后的价格路径、audit overlay state、replay-only working/position overlay、任何 live 当时不可见的诊断输出。
- 第一版触发候选：
  - `target_deterioration`: buy side 当 `working_buy_tick - target_bid_tick >= N` 时，或 sell side 当 `target_ask_tick - working_sell_tick >= N` 时，启动 side-specific toxic cooldown。
  - `cancel_requested_fill_timing`: 某侧 cancel-requested fill 后，在 `50/100/200ms` 候选窗口内阻断同侧 add-side。
  - `pending_cancel_toxic_window`: 某侧已有 cancel-requested in-flight exposure 且 target deterioration 同时成立时，阻断同侧 add-side。
- 第一版行为合同：只 block add-side。`position >= 0` 时 buy 是 add-side，可被 buy block 阻断；`position <= 0` 时 sell 是 add-side，可被 sell block 阻断；reduce-side 必须继续允许。
- 建议 helper 形状：在 `strategy_core.py` 新增类似 `adverse_timing_guard_side_blocks(...) -> GuardBlockResult`，至少包含 `buy_block`、`sell_block`、`buy_reason`、`sell_reason`。如不引入 dataclass，至少返回 `(buy_block, sell_block, buy_reason, sell_reason)`。
- 建议配置项：
  - `risk.adverse_timing_guard_enabled = false`
  - `risk.adverse_timing_guard_target_deterioration_enabled = true`
  - `risk.adverse_timing_guard_pending_cancel_enabled = true`
  - `risk.adverse_timing_guard_post_cancel_fill_enabled = true`
  - `risk.adverse_timing_guard_cooldown_ms = 100.0`
  - `risk.adverse_timing_guard_min_target_move_ticks = 2`
  - `risk.adverse_timing_guard_block_mode = "add_side_only"`
- 建议 audit 字段：
  - `adverse_timing_guard_buy_active`
  - `adverse_timing_guard_sell_active`
  - `adverse_timing_guard_buy_reason`
  - `adverse_timing_guard_sell_reason`
  - `adverse_timing_guard_buy_until_ts`
  - `adverse_timing_guard_sell_until_ts`
  - `adverse_timing_guard_target_move_ticks_buy`
  - `adverse_timing_guard_target_move_ticks_sell`
- Unit test 合同：
  - disabled 时行为完全不变。
  - target deterioration 达阈值时对应 side block。
  - cooldown 未过期时 block，过期后 unblock。
  - add-side 被 block，reduce-side 仍 allowed。
  - pending cancel + target deterioration 能触发 block。
  - helper 参数不得包含 future mid / future markout。
- Stage 6J replay 候选：
  - `baseline_inflight_only`
  - `add_side_guard_only`
  - `adverse_timing_target_deterioration_50ms`
  - `adverse_timing_target_deterioration_100ms`
  - `adverse_timing_target_deterioration_200ms`
  - `add_side_guard_plus_adverse_timing_100ms`
  - `broad_add_side_cooldown_200ms_control`
- Replay 验收合同：
  - `maker_acceptance.py` hard gates 不回退。
  - optimization replay overlays 必须是 `off/off/off`。
  - action/planned/reject/throttle match 仍为 `1.0`。
  - working semantic/blocking mismatch 为 `0`。
  - API/throttle mismatch 为 `0`。
  - strict replay lag post-startup breach/drop/fail 为 `0/0/0`。
  - source-path 必须显示目标风险下降，不能只靠减少交易改善 PnL。
  - churn、drop rate、API throttle、max abs notional 不得明显恶化。
  - 至少 `5-10-day-control-1h-06`、`5-9-small`、`5-9-noon`、`5-8-stage3-15m-livetest-v4` 四样本覆盖。
- 是否建议进入 0511T002：建议进入，但仅限 default-off 实现 + unit tests + 离线 replay。T001 不授权 live micro test，不授权默认开启规则，不授权 maker 参数搜索。

blockers：
- 无。本设计任务已完成。
- 注意：0511T002 实现前必须先 QA 通过本设计合同。

commit：
- 无

提交信息：
- 无
```
