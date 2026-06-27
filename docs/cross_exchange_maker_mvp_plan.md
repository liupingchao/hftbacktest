# Binance Lead / Hyperliquid Lag Maker MVP Plan

## 1. MVP Definition

MVP 的目标不是证明策略已经稳定盈利，也不是先建设一套完整的 Hyperliquid 单交易所回测平台。

MVP 要证明下面这条最短闭环可以成立：

```text
Binance top5 lead signal
  -> Hyperliquid decision-time fair value
  -> Hyperliquid post-only maker quote intent
  -> production-equivalent public shadow
  -> tiny-live order lifecycle and fill/cost evidence
  -> same-window replay
  -> replay/live decision-path acceptance
```

MVP 固定边界：

- 标的：`BTCUSDT` Binance USD-M lead，Hyperliquid `BTC` lag / execution venue。
- 市场输入：两边 decision-time visible top5；禁止未来数据。
- 初始预测 horizon：`1000ms`，只有经过任务验收才能调整。
- 执行：Hyperliquid `Alo` post-only，单层 maker quote，不允许 taker fallback。
- 风险：沿用当前 tiny-live 小仓位、tracked cancel、最终 `open_orders=[]`、fail-closed PnL ledger。
- MVP 不要求逐单 fill 完全复现，但要求主要决策路径一致，fill/reject/cancel/markout 分桶结果可解释且回放不乐观。

## 2. Current Reusable Baseline

不重复建设以下已经成立的部分：

- `0601T002`-`0601T005`：同步 public join、lead-lag 特征和 pricing-signal 初步证据。
- `0529T003`：Hyperliquid raw-to-npz 和 top-N provenance 基础。
- M0/M1：Hyperliquid SDK、真实 `Alo` 下单、tracked cancel、shutdown proof 和重复 canary。
- `0618T008`：fee / inventory / realized PnL fail-closed ledger。
- `0623T001`-`0624T003`：production watcher、public-state freshness、fresh-touch、anti-drift、fair-mid、edge gate 和 AWS public shadow。

当前事实：

- repaired public shadow 已经能产生 `fresh_touch_allowed_count > 0`。
- `0624T003` 中 `68` 个 fresh-touch allowed 候选里，`64` 个被 anti-drift 阻塞。
- 只有 `4` 个到达 fair-mid / edge，`edge_gate_pass_count=0`。
- 当前第一问题是 alpha/edge 的方向、时效、幅度和过滤关系，不是再次放宽 quote distance 或直接重跑 live canary。

## 3. Milestones

### M-A Signal Contract

目标：

- 证明 Binance lead 输入对 Hyperliquid 未来价格具有可重复的、decision-time-clean 的方向和幅度信息。
- 冻结 MVP v1 signal schema、horizon、freshness、方向映射和 edge 计算。

完成条件：

- 至少三个分离 public window，覆盖不同时段或市场状态。
- as-of join 无 future row。
- 样本外方向一致性和 markout 不依赖单个窗口。
- 形成一个冻结的 v1 signal contract；若证据不成立，MVP 在这里停止或回到研究。

### M-B Production-Equivalent Shadow

目标：

- live 与 replay 使用同一个 signal / fair-mid / quote-intent 内核。
- 在不下单的 public shadow 中产生足够的 would-submit 决策和反事实 markout。

完成条件：

- 多窗口 public shadow 可稳定运行。
- `signal -> fair_mid -> side -> quote -> block/would-submit` 全链审计完整。
- would-submit 样本量足够用于分桶；费用和 adverse-selection buffer 后的反事实 edge 不为系统性负值。

### M-C Minimal Hyperliquid Alignment

目标：

- 只建设 MVP 所需的 Hyperliquid 对齐层，不复制 Binance 的全部交易所假设。

完成条件：

- public market-view replay 与 live 同窗口一致。
- decision cadence、signal、side、quote intent 和 block reason 可以按稳定序号对齐。
- private lifecycle 覆盖 submit、resting、reject、cancel、cancel ack、partial/full fill。
- 真实延迟、post-only reject、fee/rebate 和 inventory/PnL 可进入 replay 校准。

### M-D Integrated MVP

目标：

- 用 tiny-live 数据标定最小执行模型，并完成 same-window replay/live 验收。

完成条件：

- 主要 action path 一致。
- replay lag 和 market-view gate 通过。
- reject/cancel/fill/markout 分桶误差在预设容差内，且 replay 不系统性乐观。
- 多个受控 tiny-live 窗口产生完整订单生命周期、费用、库存和 PnL 证据。

## 4. Sequential Task Queue

默认严格串行。只有总控明确确认无依赖时才允许并行。

### `0625T001` Public Alpha / Edge Decomposition

- 使用现有 accepted public artifacts 分解：
  - Binance lead direction / timing / magnitude
  - Hyperliquid future mid / microprice response
  - anti-drift 过滤前后差异
  - fair-mid source freshness
  - edge buffer sensitivity
- 输出结论只能是：
  - `signal_contract_candidate`
  - `needs_more_public_samples`
  - `reject_current_signal_shape`
- 不改 live 行为，不下单。

Gate:

- 若方向或时间关系错误，先修 signal，不进入 sample expansion。
- 若证据方向正确但样本不足，进入 `0625T002`。

### `0625T002` Synchronized Public Sample Expansion

- 在 `awsserver1` 收集至少三个分离窗口的 Binance top5 + Hyperliquid top5/trades。
- 每个窗口保留原始数据、session/reconnect、local receive timestamp、exchange timestamp 和 checksum。
- public-only、no-submit、no-private。
- effective-horizon gate 修复后待 QA：三窗 overlap 均约 `1800s`，完整 symmetric 1000ms context 为 `668/666/665`，但 near-target 1000ms signal-valid context 只有 `2/0/1`。
- 名义 `1000ms` 标签的有效中位时距为 `5000ms`；当前样本包不能作为 1s signal acceptance 输入。
- 修复后结论为 `needs_more_public_samples`，`t003_creation_unlocked=false`。

Gate:

- stream/join/source-age 质量通过，但 effective-horizon gate 未通过；不能创建并派发 T003。
- 后续任务 `0627T001` 先尝试 HL `l2Book fast=true` 重新采集三窗；只有 near-target 1000ms coverage 通过后，才能重新考虑 T003。
- `0627T001` 当前阻塞：接口修复和 AWS fast 证据已完成，首个正式 1800s 窗口 HL `l2Book=3335`、`l2book_fast=true`、`reconnect_count=0`。实例异常根因不是磁盘写满，而是远端 Binance alignment OOM：`binance_top5_provenance.py build-sidecars --buffer-size 10000000` 在处理 `bookTicker=1188137` 时约 `3.4G` RSS 被 kill，导致 SSH/user session 异常和后续窗口未继续；三窗 copyback / alignment / effective-horizon gate 尚未完成；T003 仍锁定。
- 新约束：`awsserver1` 只做 public raw collection，alignment 必须在 macmini 或 amdserver 上执行；后续 AWS 采集命令必须使用 `--hyperliquid-l2book-fast --skip-alignment`。

### `0625T003` Out-of-Sample Signal Acceptance

- 对 T002 多窗口运行同一个 signal runner。
- 固定 train/evaluation 边界，禁止同窗口阈值回填。
- 评估方向命中、future move、markout、basis/context、source age 和 regime stability。
- 冻结：
  - feature allowlist
  - `1000ms` horizon 或有证据的替代 horizon
  - signal normalization
  - side mapping
  - freshness limit
  - edge formula

Gate:

- 只有 `signal_contract_accepted_for_shadow` 才进入 production shadow。

### `0625T004` Shared Signal and Quote-Intent Kernel

- 从现有 watcher 中提取或整理共享的纯决策内核：
  - dual-top5 market view
  - signal
  - fair-mid
  - side
  - quote intent
  - block reason
- live shadow 和 replay 必须调用同一个内核。
- 默认 no-submit，不改变当前真实订单路径。

Gate:

- 固定 fixture 输入必须生成确定性相同输出。

### `0625T005` Multi-Window Production Shadow Acceptance

- 在多个 fresh AWS public window 上运行 production-equivalent shadow。
- 记录完整 funnel、would-submit、future markout、quote survival 和 counterfactual edge。
- 不读凭据，不调用 private/order endpoint。

Gate:

- 必须有非零且足够的 would-submit 样本。
- fee/adverse buffer 后不能呈系统性负 edge。
- 否则回到 T003/T004，不进入 tiny-live。

### `0625T006` Hyperliquid MVP Audit and Replay Contract

- 定义并实现最小统一审计：
  - run/event/decision/order identifiers
  - dual-market timestamps and source ages
  - signal/fair-mid/quote intent
  - submit/resting/reject/cancel/fill lifecycle
  - fee/inventory/PnL
- 定义 live raw artifacts、replay inputs、schema hash 和 archive contract。

Gate:

- synthetic lifecycle fixture 和现有 tiny-live artifact 都可被 validator 接受。

### `0625T007` Hyperliquid Public Market-View Replay Alignment

- 用同一 public window 重建 Binance/HL top5。
- 按 live decision cadence 重放共享决策内核。
- 对齐 market view、signal、fair-mid、side、quote intent 和 block reason。

Gate:

- future join 为 0。
- market-view/source-age/cadence gate 通过。
- action-path 差异必须有明确归因。

### `0625T008` Edge-Qualified Tiny-Live Calibration

- 仅在 T005-T007 QA 通过后创建 live execution task。
- 使用已冻结 signal contract 和当前严格风险边界运行多个微窗口。
- 只允许 `Alo`、tracked cancel、最终 open-orders proof 和 fail-closed ledger。
- 目标是获取真实 resting/reject/cancel/fill 样本，不是扩大仓位。

Gate:

- 没有完整 lifecycle/fee/inventory 证据的窗口不能算成功样本。
- 无 fill 时不得声称 PnL，可回到 maker placement/fill-acquisition 诊断。

### `0625T009` Execution Outcome Calibration

- 使用 T008 真实事件标定：
  - submit/ack latency
  - post-only reject
  - resting duration
  - cancel race
  - fill horizon
  - adverse markout
  - fee/rebate
  - inventory transition
- 形成 replay 可消费的保守执行参数。

Gate:

- 回放参数不得使用未来可见字段作为决策输入。
- 参数不足时保持保守或标记 unsupported，不允许编造 fill model。

### `0625T010` End-to-End Same-Window Replay Acceptance

- 重放 T008 tiny-live window。
- 对齐：
  - dual-market view
  - decision cadence
  - signal/fair-mid
  - side/quote intent
  - submit/cancel/reject
  - lifecycle and PnL attribution

Gate:

- action-path hard gates 通过。
- fill 不要求逐单 identity 完全一致，但分桶 fill/reject/cancel/markout 不得系统性乐观。

### `0625T011` Multi-Sample MVP Robustness

- 在至少三个 accepted live/replay window 上复核。
- 只允许窄范围、预先声明的参数选择。
- 报告 median、worst window、regime breakdown 和 inventory/cost tail。

Gate:

- 单窗口盈利不能升级为 MVP 通过。
- 最差窗口若暴露无法解释的执行或风险缺口，回到对应层修复。

### `0625T012` MVP Final Controlled Validation

- 使用冻结代码、配置、schema 和风险参数运行最终受控窗口。
- 生成一键 artifact bundle：
  - raw public/private evidence
  - live audit
  - replay audit
  - acceptance
  - lifecycle/PnL ledger
  - shutdown/open-orders proof

最终结论只能是：

- `mvp_passed_for_extended_shadow`
- `mvp_needs_targeted_repair`
- `mvp_rejected`

MVP 通过只授权更长 shadow 或下一阶段设计，不自动授权 default-on、扩仓或生产推广。

## 5. Controller Gates

总控只在 QA 最新结果通过后派下一条任务。

关键停机点：

1. T001/T003 信号无样本外价值：停止执行建设，回到 alpha research。
2. T005 would-submit edge 为负：停止 tiny-live，修 signal/quote policy。
3. T007 public replay 不对齐：停止 live，修 timestamp/market-view/cadence。
4. T008 无真实 lifecycle 或 fill：保持 M2 阻塞，禁止进入稳定 PnL 声明。
5. T010 replay 系统性乐观：停止参数优化，修执行模型。
6. T011 只在单窗口成立：保持 research，不通过 MVP。

## 6. Verification Strategy

每个任务选择最小有效验证：

- Focused tests: `python -m pytest examples/hyperliquid/<focused tests> -q`
- Syntax: `python -m py_compile <changed modules>`
- CLI: `python <runner> --help`
- Artifact checks: JSON parse、CSV schema/row count、empty-file scan、sha256。
- Public/live tasks: 明确记录 interpreter、host、commit、duration、stream counts、reconnects 和 boundary manifest。
- Repository hygiene: `git diff --check`
