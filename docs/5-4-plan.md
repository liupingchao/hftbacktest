# 2026-05-04 Live/Backtest 对齐与 Maker 策略改进计划

## 目标

最终目标不是单纯让 backtest 指标好看，而是建立一个足够可信的 maker 策略研究环境：

- live run 和 backtest 在同一实盘窗口内可解释地对齐。
- backtest 能复现实盘中的 cadence、latency、API throttle、queue/fill、inventory、order lifecycle 行为。
- 在 backtest 中改进出的 maker 策略，迁移到 live run 后能达到符合预期的 PnL、风险和成交行为。

因此下一步不应直接大规模扫描 maker 参数。正确顺序是先把仿真误差收敛到可解释，再进入策略优化。

## 当前状态

iter1 已经完成的部分：

- order lifecycle audit 已接入，能记录 submit/cancel/fill/update/expired 等生命周期事件。
- `audit_replay` 使用 `single` mode 后，cadence 对齐显著改善。
- action/reject/planned/throttle match 相比重算后的 iter0 有改善。
- position MAE、latency drop alignment 达到当前 gate。
- live run 结束时 bot position 与 REST position 一致，REST open orders 为 0。
- 没有 terminal `open_order_mismatch`。

仍未完全解决的部分：

- API/throttle guard alignment 仍是主要 H3 缺口。
- fair/reservation MAE 通过 gate，但相对重算后的 iter0 略差。
- 目前还缺少完整 PnL attribution，无法判断 backtest PnL 和 live PnL 差异来自 spread capture、fees、inventory MTM、adverse selection、latency/API miss，还是 queue/fill 模型。

第三方 review 索引见：

- `docs/5-4-review.md`

## 阶段 1：完成对齐闭环

目标：把 live/backtest 的差异压缩到可解释范围内，尤其是 API/throttle 和 PnL attribution。

### 1.1 第三方 code review

让 reviewer 先检查当前实现和归档资料：

- `examples/binance_tick_mm/audit_schema.py`
- `examples/binance_tick_mm/strategy_core.py`
- `examples/binance_tick_mm/live_tick_mm.py`
- `examples/binance_tick_mm/backtest_tick_mm.py`
- `examples/binance_tick_mm/compare_audit.py`
- `examples/binance_tick_mm/latency_from_audit.py`
- `examples/binance_tick_mm/pipeline_live_raw.py`
- iter0 资料：`baselines/iter0/align_lowfill_btcusdt_20260429_143354/`
- iter1 资料：`local_live_analysis_iter1/iter1_lifecycle_btcusdt_20260503_1800/`

review 重点：

- `audit_replay single` mode 是否实现正确。
- lifecycle event 是否足以定位 maker fill/cancel race。
- `compare_audit.py` 是否只用 `decision` 行计算主指标。
- legacy iter0 缺字段兼容是否没有掩盖真实问题。
- API/throttle mismatch 是实现问题、模型问题，还是 live 交易所机制差异。

### 1.2 修正 API/throttle guard alignment

当前主要问题是 API drop abs diff 仍未过理想 H3 标准。

需要检查：

- live 中 API interval guard、quote throttle、latency guard 的触发顺序。
- backtest 中相同 guard 的触发顺序。
- submit/cancel 是否共享同一个 API token bucket。
- live 里 REST/WS/order update 延迟是否影响下一轮 quote 决策。
- backtest 是否在同一 decision step 中模拟了过多或过少的 cancel/submit。
- throttle reason 的优先级是否和 live 一致。

产出：

- 一份 API/throttle mismatch breakdown。
- top mismatch cases 的若干条逐行解释。
- 明确哪些差异应该建模，哪些差异只能作为 residual risk。

### 1.3 建立 PnL attribution

在进入策略优化前，必须能解释 PnL。

建议拆分：

- spread capture
- maker/taker fees 或 rebate
- inventory mark-to-market
- adverse selection after fill
- latency guard missed opportunity
- API throttle missed opportunity
- cancel/fill race impact
- unfilled quote opportunity

live 和 backtest 都应输出同一套 attribution 字段或 summary。

产出：

- backtest PnL attribution summary。
- live PnL attribution summary。
- live vs backtest attribution diff。
- 能解释 PnL 差异的主要来源。

### 1.4 对齐验收标准

建议 H3+ gate：

| Metric | Target |
| --- | ---: |
| audit replay consumed ratio | `>= 0.999` |
| skipped due rows | `0` |
| max lag breaches | `0` 或有解释 |
| action match | `>= 0.97` |
| reject reason match | `>= iter1` |
| planned action match | `>= iter1` |
| throttle reason match | `>= iter1` |
| latency drop abs diff | `<= 0.02` |
| API drop abs diff | `<= 0.05` 或有明确不可建模解释 |
| position MAE | `<= 0.0008 BTC` |
| fair/reservation MAE | 低于 gate，且窗口差异可解释 |
| lifecycle evidence | 能解释全部 cancel/fill race case |

## 阶段 2：冻结 backtest research contract

目标：把回测环境定义成一个稳定的 maker 策略研究平台。

冻结内容：

- fee/funding/rebate 假设。
- latency model。
- queue/fill model。
- API budget 和 quote throttle 规则。
- order lifecycle replay/diagnostic 口径。
- initial position/open orders 注入规则。
- audit comparison 只用 `decision` 行的口径。
- PnL attribution 口径。

不冻结这些内容前，不应把参数扫描结果当成可迁移策略。

建议形成一份 contract 文档，至少说明：

- 哪些参数属于市场机制模型，不参与策略优化。
- 哪些参数属于 maker 策略参数，可以优化。
- 哪些 live/backtest 差异被认为是 residual risk。
- 每次策略优化必须使用哪些固定输入和验收指标。

## 阶段 3：Maker 策略研究与优化

目标：在可信 backtest contract 下优化 maker 策略，而不是拟合单个 live 窗口。

### 3.1 研究指标

策略选择不能只看 PnL，需要同时看：

- net PnL after fees
- max drawdown
- Sharpe 或稳定性指标
- inventory exposure
- inventory holding time
- fill rate
- maker ratio
- cancel rate
- API usage
- quote lifetime
- adverse selection after fill
- latency/API missed opportunity
- live/backtest predicted-vs-realized drift

### 3.2 参数研究方向

优先研究：

- base spread
- inventory skew
- quote size
- quote refresh cadence
- min quote lifetime
- volatility-aware spread widening
- imbalance/fair-price signal weights
- latency guard threshold
- API throttle-aware quote suppression
- adverse-selection filter
- position limit and reduce-only behavior

每个方向都要避免只优化单窗口 PnL。需要在多个 market regime 下看稳定性。

### 3.3 Walk-forward

建议流程：

1. 用 iter0、iter1 和后续 live runs 作为 calibration set。
2. 用多日历史数据做 train/test walk-forward。
3. 每个 fold 记录完整 PnL attribution。
4. 筛掉只在单日或单 regime 有效的参数。
5. 保留 PnL 稳定、inventory 可控、API usage 合理的候选。

最低要求：

- train/test 分离。
- 至少覆盖不同波动率和成交密度窗口。
- 每个候选参数都能解释收益来源。
- 参数变化不能让 API/throttle 行为超过 live 可承受范围。

## 阶段 4：Live Canary 验证

目标：验证 backtest 选出的 maker 策略是否能迁移到 live。

执行顺序：

1. 小仓位 1 小时 live canary。
2. 同窗口 backtest replay。
3. 比较 action/reject/throttle/fill/PnL attribution。
4. 若符合预期，再跑更长 live 窗口。
5. 多个 live 窗口都稳定后，才考虑扩大仓位。

live canary 验收：

- position 与 REST position 一致。
- final REST open orders 为 0。
- 无 terminal `open_order_mismatch`。
- PnL attribution 方向与 backtest 一致。
- 实际 API usage 不超过预期。
- adverse selection 没有显著超出 backtest。
- live net PnL 在 backtest 预期区间内，或偏差可解释。

## 近期执行顺序

2026-05-04 更新：iter1 已关闭为 alignment/diagnostic baseline，但不是 maker-optimization baseline。Priority1/Priority2 验收已经证明 shared API/throttle helper 不是主残差；旧 iter1 全窗口 replay 在 250ms lag gate 下 breach ratio 为 `61.55%`，因此不能直接用于 maker 参数优化。

2026-05-04 追加：working-order parity v2 把 raw order id 噪声与语义状态差异拆开。历史 iter1 样本中 semantic mismatch 为 `54356 / 62619`，identity-only mismatch 为 `8200 / 62619`；API/throttle mismatch 中 `13393 / 13692` 伴随 semantic working-order mismatch。这确认下一步不是跳过 lifecycle，而是先修 working-order replay/state parity。注意：该 v2 结果基于旧 iter1 live audit；未来 live decision 行已改为写当前 working-order snapshot，最终 parity gate 需要新 live sample。

2026-05-04 字段级修复：audit schema 已增加显式 working-order 语义字段：bid/ask qty、status、req、pending_cancel。live/backtest decision row 都会写这些字段，`compare_audit.py` 优先使用显式语义字段，旧 CSV 才 fallback 到 `local_open_orders` 字符串解析。下一段 fresh live sample 必须用这些字段做 final parity gate。

调整后的下一轮具体执行：

1. 修 working-order lifecycle replay/state parity。
2. 采集一段使用当前 audit/replay 语义的新小仓位 live sample。
3. 用 strict 250ms replay lag gate 验收新样本。
4. 设计并实现 PnL attribution。
5. 用 iter1 诊断样本和新 live sample 复核 attribution。
6. 冻结 backtest research contract。
7. 开始 maker 参数 walk-forward。
8. 选出候选后做 live canary。

## 当前不建议做的事

- 不建议直接用当前 backtest 大规模扫 maker 参数并上线。
- 不建议把旧 iter1 全窗口 replay 当成 maker 优化基线。
- 不建议只看 net PnL，不做 attribution。
- 不建议把 API/throttle mismatch 当成小问题跳过；maker 策略的真实成交和 PnL 很容易被该项主导。

当前判断：下一步应优先完成 working-order lifecycle parity、新 live sample 的 strict replay lag gate，以及 PnL attribution，然后再进入 maker 策略优化。
