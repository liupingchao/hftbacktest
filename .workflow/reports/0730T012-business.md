# 线程回报

执行线程：
- 业务线程-cross-exchange-research-design

任务ID：
- 0730T012

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0730T012.md`
- `.workflow/reports/0730T012-business.md`
- `docs/skhynix_cross_exchange_research_plan.md`
- `task_plan.md`
- `progress.md`
- `findings.md`

action：
- 审查了 T011 拉回的八段四小时 SKHYNIX campaign。
- 实测 common timeline 行数、trigger 分布、source age、segment gap、
  raw event density、price/spread 和 raw midpoint basis 分布。
- 对照现有 BTC cross-exchange signal、basis regression、maker MVP、
  collection timeline 和 maker acceptance 合约。
- 形成六层有序研究：
  - research event store
  - alignment acceptance
  - basis effectiveness
  - Binance lead / Hyperliquid lag
  - executable edge and arbitrage
  - Hyperliquid maker signal
- 定义 event-time、fixed-grid 和 response-event 三种研究视图。
- 定义 anchored walk-forward 五个 OOS folds、source-age tiers、effective
  horizon label/tolerance gate、auxiliary masks、统计门槛和失败分类。

verify：
- 本地 Conda env:
  `/Users/liu/.local/conda/envs/hftbacktest`.
- 实测 common timeline:
  - total rows: `556,861`
  - total Binance triggers: `527,660`
  - Binance depth triggers: `527,652`
  - Binance snapshot triggers: `8`
  - Hyperliquid fast-L2 triggers: `26,519`
  - Hyperliquid standard-L2 triggers: `2,682`
- 实测 raw rate:
  - Binance bookTicker: approximately `390.76/s`
  - Binance trades: approximately `355.40/s`
  - Hyperliquid BBO: approximately `9.92/s`
  - Hyperliquid fast L2: approximately `1.84/s`
  - Hyperliquid trades: approximately `4.65/s`
- Hyperliquid fast-L2 age at Binance triggers:
  - p50 approximately `271.66ms`
  - p99 approximately `628.33ms`
  - `90.11% <= 500ms`
- Hyperliquid standard-L2 age at Binance triggers:
  - p50 approximately `2681.34ms`
  - p99 approximately `5358.82ms`
- Segment restart gaps:
  - approximately `1080.12-1433.69ms`
- Raw midpoint basis:
  - median approximately `13.21bps`
  - p99 approximately `48.40bps`
  - range approximately `-33.83bps` to `76.67bps`
- Confirmed the current registry labels SKHYNIX basis diagnostic-only.
- Markdown structure and `git diff --check` are part of final verification.

done：
- 研究方案已形成：
  `docs/skhynix_cross_exchange_research_plan.md`.
- 用户指定的 alignment、basis、lead-lag、套利、Binance lead 和
  Hyperliquid maker signal 均有独立实验与验收门槛。
- 方案明确禁止把预测 basis 直接等同于套利，也禁止 exact fill/PnL 声明。
- 第七步只能建议补采；新增 AWS/public-data 采集必须另建任务、获得用户
  明确授权，并且只在用户确认的活跃交易时段启动。

blockers：
- 无文档设计阻塞。
- T011 独立 QA 仍未完成，因此正式研究实现应继续保留 source-status caveat。
- 四小时单日数据最多支持研究候选或拒绝，不能支持生产 promotion。

commit：
- 无

提交信息：
- 无
