# Binance Lead / Hyperliquid Lag Maker MVP Plan

## 2026-07-28 Status Reconciliation

This section is the current-status overlay for the historical plan below.
Task and QA files remain the detailed source of truth.

- M-A Signal Contract: complete. `0625T003` QA passed with the accepted
  `binance_lead_composite` contract.
- M-B Production-Equivalent Shadow: complete. `0625T004` and `0625T005`
  passed; later basis-regression and exact seeded-dynamic public-shadow work
  passed through `0722T063`, `0722T066`, and `0722T067`.
- M-C Minimal Hyperliquid Alignment: mechanism/evidence scope complete.
  Public replay reproduced `10704/10704` decisions; fixed single-level
  same-window mechanism acceptance passed through `0721T046`.
- M-D Integrated MVP: incomplete. `0726T068` proved exact seeded-dynamic
  submit/reject/resting/cancel/account/terminal behavior, but produced zero
  fills and zero liquidity-role rows.

Current hard boundary:

- role-known fill, fee/rebate, fill-rate, markout/PnL, maker viability,
  promotion, and final MVP remain blocked;
- historical text below that says T003 is pending or T004-T007 are not created
  is superseded by this reconciliation and
  `docs/cross_exchange_mvp_task_classification.md`.

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

分支和事实源约束：

- `cross-exchange` 是 Binance-lead / Hyperliquid-lag MVP 的唯一正式开发和 workflow 事实源分支。
- 其他分支只作为临时、备份或恢复分支；其中的任务只有在 task file、report、artifact 和必要代码恢复到 `cross-exchange` 后，才可作为正式前置事实。
- M-A / M-B / M-C / M-D 四个里程碑是本分支的最高规划约束；后续任务不得绕过前置 milestone gate。
- 历史任务分类见 `docs/cross_exchange_mvp_task_classification.md`。

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
- 当前第一问题已经从早期 alpha/edge 决策性推进到最小 live/replay 校准边界：`0625T003`-`0625T007` 已通过 QA，`0706T002 / 0625T008` 完成一单真实 submit/resting/cancel/open-orders 校准，`0706T003 / 0625T009` 只接受该一单 artifact 的保守执行结果校准。
- `0706T003` 直接支持的事实仅限：精确一单 envelope 的 submit endpoint 可达、Hyperliquid `Alo` post-only、order response 为 `resting`、primary tracked cancel 成功、独立最终 open-orders count 为 `0`。
- `0706T003` 明确不支持：submit/ack latency、resting duration、cancel latency、cancel-fill race、fill horizon/probability、fee/rebate、inventory transition、realized PnL、stable PnL、maker viability。
- 因此后续路线必须拆成两条：先做不新增 live 行为的 supported-fact same-window replay sanity/acceptance；完整 fill/cost/PnL replay、multi-sample robustness 和最终 MVP validation 仍需要新的 formal task、真实证据和显式 live 授权。
- `0706T007 / 0625T010-LIVE-EVIDENCE-ACQUISITION` 已在显式授权后执行一次最小 live evidence window，但该窗口没有 eligible fresh-touch candidate，因此没有调用 order endpoint、没有调用 cancel endpoint、没有 fill，最终 open-orders 为 `0`。该结果是安全的 fail-closed live evidence，不是 full `0625T010` 通过。

## 2.1 Current Roadmap Boundary After `0706T003`

当前 `0625T009` 不是完整执行模型通过，只是把一单真实 artifact 转成 replay 可消费的窄事实。

已完成的窄验收：

- `0706T005 / 0625T010-SCOPED` 已通过 QA。
- replay 已能保守表达一单 submit/resting/cancel/final-open-orders 事实。
- unsupported execution/economics/PnL/viability 字段保持 fail-closed。
- 该结果不解锁 `0625T011`、完整 `0625T010`、新 live-submit、repeated-window、fill-seeking run、稳定 PnL 或 maker viability 声明。

已完成的 scoped 任务边界：

- `0625T010-SCOPED` / supported-fact same-window replay acceptance。
- 输入只能是 `0706T002` 和 `0706T003` 已验收 artifact。
- 验收目标只允许覆盖 submit intent、post-only/resting response、tracked cancel、final open-orders proof，以及 unsupported 字段 fail-closed。
- 不得声称 fill model、fee/rebate、inventory/PnL、稳定收益或 maker viability。

必须暂停并需要新授权的路线：

- 任意新的 live-submit、repeated-window、fill-seeking、closer-to-market placement、size/quote envelope 变化。
- 完整 `0625T010` end-to-end lifecycle/PnL replay。
- `0625T011` multi-sample robustness。
- `0625T012` final controlled validation。

已完成的 full T010 preflight 和一次授权 live evidence attempt：

- `0706T006 / 0625T010-FULL-PREFLIGHT` 已通过 QA。
- full `0625T010` 已推进到 no-submit evidence-acquisition preflight。
- `0706T007 / 0625T010-LIVE-EVIDENCE-ACQUISITION` 已通过 QA，结论是 `full_t010_live_evidence_blocked_no_order_submitted`。
- 授权 envelope 内观测到 public flow 和 `10` 个 fresh-touch candidates，但 `fresh_touch_allowed_candidate_count=0`，所以提交前 fail-closed。
- `real_order_endpoint_called=false`，`real_cancel_endpoint_called=false`，`fill_count=0`，window 与 independent final open-orders count 均为 `0`。
- 完整 same-window live evidence、cross-exchange decision path、submitted lifecycle、latency/ordering、fill/no-fill economics、fee/rebate、inventory、PnL ledger 仍缺失。
- 下一步不能自动继续 full `0625T010`。若继续推进，必须另建 formal task：要么把已验收的 cross-exchange signal/fair-mid kernel 接入 live decision path 后再收集证据，要么显式授权新的 evidence envelope。

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
- `0627T001` 曾经阻塞：接口修复和 AWS fast 证据已完成，首个正式 1800s 窗口 HL `l2Book=3335`、`l2book_fast=true`、`reconnect_count=0`。实例异常根因不是磁盘写满，而是远端 Binance alignment OOM：`binance_top5_provenance.py build-sidecars --buffer-size 10000000` 在处理 `bookTicker=1188137` 时约 `3.4G` RSS 被 kill，导致 SSH/user session 异常和后续窗口未继续。
- 新约束：`awsserver1` 只做 public raw collection，alignment 必须在 macmini 或 amdserver 上执行；后续 AWS 采集命令必须使用 `--hyperliquid-l2book-fast --skip-alignment`。
- `0627T001` QA 已通过：三窗 HL `l2Book=3335/3324/3326`，完整 symmetric 1000ms contexts `10745`，near-target 1000ms signal-valid contexts `10704`，recommendation=`sample_contract_ready_for_signal_acceptance`，`t003_creation_unlocked=true` 仅表示可由总控创建/派发 T003。
- `0625T003` business execution 已完成并待 QA；业务建议为 `signal_contract_accepted_for_shadow`，但后续 M-B 任务仍需等待 QA 通过和总控明确派发。

### `0625T003` Out-of-Sample Signal Acceptance

- 当前状态：业务执行完成，`.workflow/tasks/0625T003.md` 为 `待验收`；QA 通过前不得派发 M-B 后续任务。
- 输入只能使用 `0627T001` QA-accepted HL fast package；不得使用 repaired-but-invalid ordinary T002 package 作为 signal acceptance rows。
- 对三段 public window 运行同一个 signal acceptance runner。
- 只允许 nominal `1000ms` 且 row-level effective horizon valid 的 label：
  - `1000ms <= effective_future_age_ms <= 1250ms`
- 固定 train/evaluation 边界，默认 leave-one-window-out；禁止同窗口阈值回填、side mapping 回填或开放式 feature search。
- 评估方向命中、future move、markout、basis/context、source age、regime stability、side mapping stability 和 fee/adverse-buffer-adjusted edge proxy。
- 候选 feature 只能来自 decision-time-visible allowlist。
- 若通过，冻结：
  - feature allowlist
  - `1000ms` horizon 和 effective-horizon row condition
  - signal normalization
  - side mapping
  - freshness limit
  - edge formula
- 业务执行结果：
  - package: `local_live_analysis/cross_exchange_mvp_signal_acceptance_0625T003/`
  - recommendation: `signal_contract_accepted_for_shadow`
  - accepted candidate: `binance_lead_composite`
  - threshold: `abs(z) >= 1.0`
  - side mapping: `positive_signal_buy_negative_signal_sell`
  - caveat: one source-age bucket has negative adjusted proxy and must remain visible in QA / shadow design.

Gate:

- Final recommendation 只能是 `signal_contract_accepted_for_shadow`、`needs_more_samples` 或 `reject_current_signal_shape`。
- 只有 `signal_contract_accepted_for_shadow` 才能解锁后续 production-equivalent shadow task。
- T003 不授权 watcher change、private/order endpoint、live order、canary、promotion 或 Hyperliquid replay/alignment 结论。

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
- 当前已验收事实：`0706T002` 在显式授权后只完成了一单 BTC `Alo` post-only canary，得到 submit/resting/primary cancel/final open-orders `0` 证据。
- 当前未获得完整 repeated-window、reject、fill、fee、inventory 或 realized PnL 证据。

Gate:

- 没有完整 lifecycle/fee/inventory 证据的窗口不能算成功样本。
- 无 fill 时不得声称 PnL，可回到 maker placement/fill-acquisition 诊断。
- 任何新增 live-submit、repeated-window 或 fill-seeking T008 后续任务都必须重新建正式任务并获得显式授权。

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
- 当前已验收事实：`0706T003` 只基于 `0706T002` 一单 artifact 生成保守校准。
- 当前 supported 参数只覆盖 submit endpoint reachability、post-only `Alo`、`resting` response、primary tracked cancel success、final open-orders count `0`。
- 当前 unsupported 参数必须在后续 replay 中 fail-closed：submit/ack latency、resting duration、cancel latency、cancel-fill race、fill horizon/probability、fee/rebate、inventory transition、realized PnL、stable PnL、maker viability。

Gate:

- 回放参数不得使用未来可见字段作为决策输入。
- 参数不足时保持保守或标记 unsupported，不允许编造 fill model。

### `0625T010-SCOPED` Supported-Fact Same-Window Replay Acceptance

- 当前状态：`0706T005` QA 已通过。
- 重放 `0706T002 / 0706T003` 支持的一单事实窗口。
- 对齐范围只包括：
  - order intent / submit path
  - post-only `Alo` / `resting` response
  - primary tracked cancel
  - independent final open-orders proof
  - unsupported lifecycle/economics/PnL 字段 fail-closed

Gate:

- 不新增 live/private/order 行为。
- 不使用 unsupported 字段生成乐观 fill/cost/PnL。
- replay 若不能表达 unsupported/fail-closed 边界，则不得进入完整 T010 或 T011。
- 通过后只能说明 supported-fact replay sanity 通过，不解锁 multi-sample MVP robustness。

### `0625T010` Full End-to-End Same-Window Replay Acceptance

- 重放 T008 tiny-live window。
- 对齐：
  - dual-market view
  - decision cadence
  - signal/fair-mid
  - side/quote intent
  - submit/cancel/reject
  - lifecycle and PnL attribution

Gate:

- 需要新的完整 T008-style live evidence：market view、decision path、submit/cancel/reject/fill、fee/rebate、inventory/PnL attribution。
- action-path hard gates 通过。
- fill 不要求逐单 identity 完全一致，但分桶 fill/reject/cancel/markout 不得系统性乐观。
- 当前 `0706T003` 证据不足以启动完整 T010；必须先获得新 live evidence 和显式授权。

### `0625T011` Multi-Sample MVP Robustness

- 在至少三个 accepted live/replay window 上复核。
- 只允许窄范围、预先声明的参数选择。
- 报告 median、worst window、regime breakdown 和 inventory/cost tail。

Gate:

- 单窗口盈利不能升级为 MVP 通过。
- 最差窗口若暴露无法解释的执行或风险缺口，回到对应层修复。
- scoped T010 通过不等于 T011 前置通过；T011 只接受完整 T010 通过后的多个 live/replay window。

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
