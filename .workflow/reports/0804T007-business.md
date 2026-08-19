# 线程回报

执行线程：
- 业务线程-python/cross-session-research-design

任务ID：
- 0804T007

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 第五轮独立 QA 已通过，P0-P3 均为 0。

files：
- `docs/skhynix_three_session_commonality_research_plan.md`
- `.workflow/tasks/0804T007.md`
- `.workflow/reports/0804T007-business.md`
- `.workflow/reports/0804T007-qa-round1.md`
- `.workflow/reports/0804T007-qa-round2.md`
- `.workflow/reports/0804T007-qa-round3.md`
- `.workflow/reports/0804T007-qa-round4.md`
- `.workflow/reports/0804T007-qa.md`

action：
- 增加 `d_bh = Binance bid1 - Hyperliquid ask1` 和
  `d_hb = Hyperliquid bid1 - Binance ask1` 的
  native/common-quote/bps/tick 定义。
- 增加双边 spread 守恒恒等式、strict as-of BBO decision population、
  source-age 和 quantity multiplier 门禁。
- 设计 level、100ms innovation、z-score excursion、persistence episode
  和 near-crossing control。
- 设计 10ms-2s future impact、signal survival、first mover、time-to-zero、
  half-closure 和 maximum-widening 输出。
- 分解 Binance/Hyperliquid 两条 quote leg 对形成与闭合的贡献，并定义
  Binance-driven、Hyperliquid-driven 和 mixed。
- 增加 H-ASK/H-BID maker protection hypotheses，以及 fee/latency/capacity
  sensitivity 边界。
- 增加 quote-currency as-of conversion 和 contract multiplier provenance；
  稳定币 1:1 只能作为显式 scenario。
- 将信号纳入三 session C1-C3、lag-shift、bootstrap、negative controls、
  deliverables 和 acceptance gates。
- 将唯一确认性 maker outcome 固定为 Hyperliquid 对应 BBO repricing；
  depth/trade/replenishment 进入独立校正的 secondary family，不能提升 tier。
- Closure/formation 全部使用 common-quote `Q`，并冻结 `t`/`t+h`
  conversion as-of 规则。
- 冻结 100ms formation window、两个方向四条 leg contribution 公式，以及
  2s scan limit/right-censor/Kaplan-Meier 路径合同。
- 冻结 level/change trailing robust-z、joint OLS beta、连续/事件 C1 支持、
  单侧 null、实际效应门槛和 `100/250/500/1000/2000ms` BBO 主 family。
- 分别定义 sell-leg ticks、buy-leg ticks 和 conservative common ticks。
- 冻结 fast-L2 anchor-price depletion、方向一致 aggressor trade arrival、
  50%/80% replenishment-failure 标签及可观测性/censoring。
- 冻结 Jul30 `level_z/change_z` 0.5%-99.5% nearest-rank OOD envelope。
- 冻结同方向/同事件类 500ms refractory/merge，state false-to-true 和
  point-event immediate-exit 去重语义。
- BBO bootstrap 按 `decision_ts` 归块，每个 draw 复制整块 union rows，
  并对 30 个 joint fit keys 逐一重建 eligibility 和 OLS，再映射为
  60 个 level/change hypothesis keys。

verify：
- 两个方向差值之和严格等于两边负 spread 的定义已写入 row-level gate。
- Closure 与两条 venue-leg contribution 的守恒关系已写入验收门禁。
- 数量只有转换为 base-equivalent 后才能跨 venue 比较。
- `1/2/5ms` 仅定义为 observed-state survival，不声称真实 exchange
  reaction resolution。
- Aug04 freeze/first-read ledger 已覆盖 directional-BBO builder。
- 三个目标文件建立 scoped intent-to-add Git baseline 后，
  `git diff --check -- <targets>` 通过。
- Structural family 的 1s/2s 与 directional-BBO 的五个主 horizon 已明确
  分开校正，不能互相替代。
- Lag-shift surrogate 重建 BBO state/predictor/outcome 并重新拟合完整 OLS，
  不只打乱最终 beta 表。

done：
- 三数据集研究方案已包含两个方向性 BBO 差值的独立研究层，可拆分为
  后续实现任务。
- 本任务未运行研究管线或读取新的 outcome，不形成 exact fill、
  executable arbitrage 或 PnL 结论。

blockers：
- 无

commit：
- 无

提交信息：
- 无
