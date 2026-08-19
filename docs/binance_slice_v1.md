可以。先把这条 slice 的目标收紧一下：

[
\boxed{
\text{可信地识别一次队列冲击}
\rightarrow
\text{刻画冲击后的条件响应}
\rightarrow
\text{尽早判断 absorption / continuation}
\rightarrow
\text{决定自己的 maker order 是保留、撤销还是后移}
}
]

第一版不要直接上 motif、HMM 或完整 GLFT 参数重估。先完成一条能够从 Tardis 原始数据一直跑到 hftbacktest maker action 的闭环。

---

# 一、先确定默认研究配置

我建议采用下面的默认配置启动：

| 项目              | 第一版选择                                                         |
| --------------- | ------------------------------------------------------------- |
| 标的              | **BTC 线性永续合约**                                                |
| Primary venue   | 先做 3 个候选交易所的数据质量审计，再选一个；默认候选是 Binance USDS-M `BTCUSDT`        |
| Secondary venue | OKX BTC-USDT-SWAP 或你未来真实交易的第二 venue                           |
| Primary trigger | (I_{\text{best}}\ge 0.5)                                      |
| 时间基准            | **local timestamp 为在线决策主时钟**，exchange timestamp 用于机制核对        |
| 深度              | 重建完整 L2；研究特征主要用前 5 档，前 25 档用于验证与外围流动性                         |
| 工程数据            | 3 个独立交易日，只用于开发和人工检查                                           |
| 正式研究数据          | 60 个连续交易日                                                     |
| 时间切分            | 30 日 discovery / 15 日 calibration / 15 日 untouched final test |
| 响应窗口            | 同时保存前 50、后 100 个市场事件，以及真实时间窗口                                 |
| 初始 maker action | `KEEP`、`CANCEL`、`WIDEN_ONE_TICK` 三种                           |
| 回测框架            | hftbacktest，但核心 detector 与 policy 不能写死在 hftbacktest 内部        |

为什么先 BTC，不先 ETH：这不是因为假设 BTC 的规律更“真”，而是第一条 slice 首先要压力测试数据重建、trade–book 同步、事件数量和执行模拟。之后用 ETH 做资产外推检验。

为什么不直接固定 Binance：这条研究非常依赖盘口 feed cadence。一个 venue 的 full-depth 更新如果太慢，你就无法可信地研究 5ms 或 10ms 的 refill；因此先用少量数据做 venue bake-off，比拍脑袋选交易所靠谱得多。

---

# 二、你现有的 Tardis 数据分别怎么用

| 数据类型                  | 第一版中的角色                                         | 是否进入 trigger |
| --------------------- | ----------------------------------------------- | -----------: |
| `incremental_book_L2` | **盘口重建的主数据源**；计算 pre-trigger queue、OFI、深度变化     |            是 |
| `trades`              | aggressor side、成交价格、成交量、trade burst             |            是 |
| `book_snapshot_25`    | 重建正确性的 oracle；图形检查；不能同时作为主数据源                   |            否 |
| `book_snapshot_5`     | 快速可视化或轻量 QA；有 25 后基本冗余                          |            否 |
| `book_ticker`         | 交易所原生 BBO；检查 L2 BBO 是否延迟，必要时建立独立的 fast-BBO view |         条件使用 |
| `quotes`              | 从 L2 重建出的 BBO，用于回归测试，不是独立证据                     |            否 |
| `derivative_ticker`   | funding、OI、mark/index basis 等 context tag       |       第一版不使用 |
| `liquidations`        | 标记 episode 是否与强平重叠；以后作为独立 anchor                |       第一版不使用 |

Tardis 的 `incremental_book_L2.amount` 是该价格档更新后的**绝对数量**，不是 delta；同一个 WebSocket 消息里的多个档位更新可通过相同 `local_timestamp` 分组。初始 snapshot 前的非 snapshot 更新要跳过；重新出现 snapshot 时必须丢弃旧盘口。`book_snapshot_25` 和 `book_snapshot_5` 则是基于 L2 重建、每次相关档位变化时生成的完整 top-N 快照。`trades.side` 表示 taker/aggressor side。`quotes` 来自 L2，而 `book_ticker` 来自交易所原生 BBO channel，因此两者不能被当成两个独立市场观测。([Tardis Documentation][1])

这里最重要的一条纪律是：

> `incremental_book_L2 + trades` 负责研究；
> `book_snapshot_25 + quotes + book_ticker` 负责审计和发现不一致。

不要拿 `book_snapshot_25` 做主盘口，然后再用它证明盘口重建正确，那会变成一种圆润的自我表扬。

---

# 三、先把 trigger 写成严格的事件契约

## 1. 基础定义

对于一次主动买入冲击：

[
p_0=P^{ask}_{t_s^-}
]

[
Q_0=Q^{ask}_{t_s^-}(p_0)
]

其中 (t_s) 是当前 aggressive buy burst 的第一笔成交到达本地的时刻，(Q_0) 是**在看到第一笔成交之前**的最优卖盘数量。

对于主动卖出，使用最优买盘，逻辑镜像。

第一版建议把原始公式收紧成一个 clean trigger：

[
I_{\text{best}}
===============

\frac{
\sum_{j\in B}
v_j\mathbf 1(p_j=p_0)
}{
Q_0
}
]

当：

[
I_{\text{best}}\ge \theta,\qquad \theta=0.5
]

第一次成立时，发出 trigger。

也就是说，第一版研究的是：

> 同一个 aggressive burst 在原来的最优价格上，成交量至少达到 trigger 前显示队列的 50%。

同时再保存一个较宽松变量：

[
I_{\text{burst}}
================

\frac{\sum_{j\in B}v_j}{Q_0}
]

它包括 burst 向更深档位扫过的成交量，但不用于第一版主 trigger。这样可以把“明确消耗 best queue”和“总体 aggressive flow 很大”区分开。

此前文档建议第一版只研究 best queue 被主动成交消耗 30%～50% 的单一 trigger，而不是同时混合多个事件类型。

---

## 2. trade burst 的定义

第一版按下面的 state machine 实现：

1. 第一笔 trade 的 aggressor side 已知；
2. 第一笔 trade price 必须等于当时看到的 passive best (p_0)；
3. 连续相同方向的 trades 归入同一 burst；
4. 出现以下任一条件时 burst 结束：

   * 出现相反方向 trade；
   * inter-trade gap 超过 (g)；
   * 达到最大 burst duration；
   * snapshot reset 或数据断点；
5. 当累计 (I_{\text{best}}) 第一次越过 0.5 时立即发出事件，不等待 burst 结束；
6. 每个 burst 只产生一个 primary trigger。

(g) 不用 outcome 或 PnL 调优。先在 engineering days 上观察同方向连续成交的 inter-arrival distribution，选一个能分隔 exchange packet/microburst 与普通连续流的阈值，随后冻结。可以从 1ms 作为初始实现值，同时保留：

* exact-timestamp packet；
* (g=250\mu s)；
* (g=1ms)；

三种定义做稳定性检查。

---

## 3. local-time 与 exchange-time 两个版本

为保证它最终可在线执行，主 trigger 使用：

[
Q_0^{local}
===========

\text{看到 trade 前，本地已经收到的最新盘口}
]

同时离线计算：

[
Q_0^{exchange}
==============

\text{按 exchange timestamp 重排后的盘口}
]

两者的用途不同：

* local-time trigger：回答“live maker 当时知道什么”；
* exchange-time trigger：回答“撮合层面可能发生了什么”。

如果两者对同一个 episode 的 (p_0)、(Q_0) 或是否越过阈值判断不同，给 episode 打上：

```text
ordering_ambiguous = true
```

第一版主分析优先使用两种时钟下均成立的 robust trigger。

Tardis CSV 在各自文件内保留捕获顺序，并以 `local_timestamp` 切分和排序；但 trades 与 L2 已经被拆成不同文件后，同一个微秒上的跨文件先后关系不应凭空假设。因此必须保留 tie ambiguity，而不是偷偷规定“trade 永远先于 book”或反过来。([Tardis Documentation][2])

---

## 4. 事件排除条件

以下情况不生成 clean trigger：

* 尚未收到有效初始 snapshot；
* trade side 为 `unknown`；
* (Q_0\le 0)；
* 第一笔 trade price 与本地 best 不一致；
* spread 无效、crossed book 未解决；
* snapshot reset 或数据 gap 落在 pre-window；
* 同 timestamp 排序会改变是否触发；
* duplicate trade ID；
* episode 与上一个 trigger 属于同一个 shock cluster。

但不要直接把强平、funding 附近或高波动 episode 删除。先做 tag：

```text
liquidation_overlap
funding_window
high_volatility
multiple_shock_contamination
```

之后分别报告 clean cohort 和 full cohort。否则很容易通过“清洗”把真正重要的市场状态洗掉。

---

# 四、整个 slice 的里程碑顺序

| 节点  | 要回答的问题                  | 核心交付物                  | 通过标准                     |
| --- | ----------------------- | ---------------------- | ------------------------ |
| M0  | 研究对象是否定义清楚？             | `research_contract.md` | 无歧义、无未来数据                |
| M1  | 哪个 venue 的数据支持这个研究？     | venue audit report     | 确定主 venue 和最小可观察 horizon |
| M2  | 能否准确重建盘口？               | canonical replay + QA  | 与 snapshot/quotes 高度一致   |
| M3  | trigger 是否真的被正确识别？      | `triggers.parquet`     | 人工检查和顺序鲁棒性通过             |
| M4  | 能否形成可复现案例库？             | episode store          | 任意 episode 可重新回放         |
| M5  | 是否存在可解释的响应事实？           | event-study report     | 跨日、条件化后仍稳定               |
| M6  | 能否得到稳定 response state？  | frozen taxonomy/model  | 不使用 future outcome 聚类    |
| M7  | 在 untouched days 上成立吗？  | final test report      | 原型、概率和 outcome 均可复现      |
| M8  | 能否足够早地识别状态？             | online prefix detector | 决策早于主要价格结果               |
| M9  | 能否改善自己的 maker 决策？       | GLFT overlay backtest  | queue/latency 敏感性下仍有效    |
| M10 | replay、paper、live 是否对齐？ | parity + drift monitor | 同一输入产生同一决策               |

下面逐个展开。

---

# M0：冻结研究契约

## 目标

在写复杂代码前，把“什么算一次事件”冻结下来。

建议固定：

```text
asset                 = BTC linear perpetual
primary_theta         = 0.50
sensitivity_theta     = [0.25, 0.75, 1.00]
trigger_clock         = local arrival time
primary_volume        = trades executed at pre-trigger best
one_trigger_per_burst = true
direction_normalized  = true
outcome_in_discovery  = false
```

需要同时写三条 primary hypothesis：

### H1：冲击剂量与响应

[
I_{\text{best}}\uparrow
\Rightarrow
P(\text{best depleted before refill})\uparrow
]

### H2：条件 response

在相似 pre-state 下，快速 aggregate replenishment 与较弱的同方向价格延续相关。

### H3：maker action

在可在线识别的 continuation state 中撤掉危险一侧的 maker quote，能降低 adverse selection；在 absorption state 中保留 quote，能够避免过度撤单造成的 fill 损失。

这里的“replenishment”是聚合价格档位补充，不是某一张具体订单回来了。普通 L2 把多个参与者聚合在价位上，无法知道是否是同一个 maker，也不能观测真实 queue position 或订单生命周期。

---

# M1：先做 venue bake-off，不要立即跑 60 天数据

## 输入

同一个 BTC 线性永续，在每个候选 venue 上选择相同的 3 个 engineering days。

最好覆盖：

* 一个普通交易日；
* 一个高波动交易日；
* 一个相对平静交易日。

这些日期只用于工程，不进入正式统计结果。

## 每个 venue 输出以下指标

### 数据质量

* snapshot resets/day；
* sequence 或数据断点数量；
* negative feed latency 比例；
* trade side unknown 比例；
* duplicate trade ID；
* crossed/locked book 比例；
* trade price 落在 reconstructed spread 外的比例；
* 同 timestamp 的 trade/book ambiguity 比例。

### 可观察性

* incremental L2 message inter-arrival P50/P90/P99；
* book_ticker inter-arrival P50/P90/P99；
* trade 后首次 L2 update latency；
* trade 后首次 BBO update latency；
* 在 1、5、10、50、100ms 内能够看到 book response 的 episode 比例；
* (I_{\text{best}}\ge0.5) 的事件数量/day；
* median pre-trigger queue 和事件严重程度分布。

定义一个最小可观察 horizon：

[
h_{\min}
========

\min
\left{
h:
P(\text{next relevant book update}\le h)\ge 80%
\right}
]

如果某 venue 的 (h_{\min}=100ms)，就不要在后面自欺欺人地研究 5ms full-depth refill。可以：

* 用 native `book_ticker` 做 BBO-only 早期特征；
* 或选择 L2 cadence 更合适的 venue。

HftBacktest 提供了 Tardis `convert_fuse`，可以把 depth 与 `book_ticker` 结合，以获得更细的 BBO 更新；但官方文档也明确提醒，异步 feed 的 exchange timestamp 与 local arrival 顺序可能产生轻微不一致，因此 fused view 必须单独做一致性审计。([HftBacktest][3])

## M1 的人类决策

基于 audit 选定：

1. primary venue；
2. 使用 `L2-only` 还是 `L2 + independent fast-BBO view`；
3. 第一版真正能研究的最短 horizon。

默认优先级是：

[
\text{未来实际部署 venue}

>

\text{数据语义最清楚的 venue}

>

\text{事件数量最多的 venue}
]

---

# M2：构建 canonical replay 与盘口重建器

不要把 hftbacktest 的 NPZ 当作唯一事实源。建议维护两个产物：

## 1. Canonical research stream

统一事件结构：

```text
event_id
venue
symbol
source                 # depth / trade / book_ticker
exchange_timestamp
local_timestamp
source_sequence
event_type
side
price_tick
quantity
trade_id
is_snapshot
snapshot_epoch
data_quality_flags
```

所有价格先转成 integer ticks，避免浮点价格比较。

## 2. HftBacktest event stream

从 canonical stream 派生，供 queue、latency 和 fill simulation 使用。

这样做可以避免研究逻辑被 hftbacktest 的内部 event flags、timestamp correction 或 snapshot handling 绑死。

---

## 盘口重建规则

1. 等待第一个 snapshot；
2. snapshot 内逐层设置绝对 amount；
3. amount = 0 时删除该价格层；
4. 新 snapshot epoch 到来时清空旧 book；
5. 同一 `local_timestamp` 的多档位更新视为同一 exchange message batch；
6. 保留更新前和更新后的 book state；
7. 分别维护：

   * fixed price map；
   * top-25 view；
   * BBO view；
   * snapshot epoch。

## 验证方法

使用 `book_snapshot_25` 作为 oracle：

* best bid/ask price exact match；
* best bid/ask amount exact match；
* top-25 price exact match；
* top-25 amount exact match；
* snapshot reset 后一个 message 内恢复一致。

Tardis 的 snapshot 数据会移除 crossed levels，所以验证器必须以相同规则生成 comparison view，否则会制造伪 mismatch。([Tardis Documentation][1])

建议 gate：

```text
BBO exact match                  > 99.999%
Top-25 exact match               > 99.99%
persistent divergence            = 0
unexplained snapshot reset       = 0
invalid/crossed final book       = 0
```

这里真正重要的不是追求一个漂亮百分数，而是：

> 每一个 mismatch 都必须能归因于 snapshot reset、cross removal、timestamp tie 或数据缺口。

## Binance 时间戳特别检查

HftBacktest 的 Tardis converter 提醒，Binance Futures depth 数据使用发送事件时间 `E`，而非撮合发生时间 `T`，因此从这些字段计算出的 feed latency 可能略微低估真实值。这个差异不能直接拿来作为真实 live latency 模型。([HftBacktest][3])

---

# M3：实现 trigger detector，并进行人工验收

输出表建议至少包含：

```text
episode_id
shock_cluster_id
venue
symbol
direction
burst_start_local_ts
threshold_cross_local_ts
p0_tick
q0
volume_at_p0
total_burst_volume
impact_best
impact_burst
burst_duration_us
levels_swept
mid0_tick
spread0_tick
snapshot_epoch
ordering_ambiguous
book_trade_mismatch
liquidation_overlap
funding_window
```

## 必做单元测试

### No-lookahead test

删除 trigger 后所有数据，事件识别结果必须完全不变。

### Side mirror test

将 buy/ask episode 镜像成 sell/bid，结果应保持对称。

### Threshold monotonicity

如果同一 burst 越过 0.75，就必然已经越过 0.5。

### Snapshot reset test

不能跨 snapshot epoch 累积 trade burst。

### Tie-order sensitivity

同 timestamp 下分别采用：

* depth-first；
* trade-first；

两种处理，如果事件判断不同，必须标记 ambiguous。

### Duplicate test

同一 trade ID 不得重复累计；没有 trade ID 时，不得仅因为 price、amount、timestamp 相同就武断删除，因为可能是两笔真实成交。

## 人工验收

按以下 strata 抽取至少 200 个 episode：

* buy / sell；
* (I\in[0.5,0.75))；
* (I\in[0.75,1))；
* (I\ge1)；
* best survives / disappears；
* 高/低 pre-trigger imbalance；
* ordering robust / ambiguous。

每个 episode 输出一张统一图：

```text
top-5 depth heatmap
best queue path
aggressive trades
cumulative impact fraction
book_ticker BBO
mid / microprice
snapshot reset and data flags
```

M3 的通过标准不是“有很多事件”，而是：

> 随机拿出任意一个 episode，都能解释为什么 (Q_0) 是这个数量、哪些 trades 被纳入 burst、阈值在哪一笔成交上越过。

这一步不过，后面所有 motif 和 PnL 都是在沙滩上盖塔。

---

# M4：建立 case library

每个 episode 拆成四个互不混淆的部分：

[
E_i=
\left(
S_i^{pre},
T_i,
R_i,
O_i
\right)
]

其中：

* (S^{pre})：trigger 前市场状态；
* (T)：trigger 的严重程度和具体结构；
* (R)：冲击后的订单簿响应；
* (O)：未来价格和执行结果。

**Outcome 不能进入 response motif 或 state discovery。**

---

## Pre-state (S^{pre})

第一版建议保留：

### 盘口状态

* spread ticks；
* L1 queue imbalance；
* L1–L5 depth imbalance；
* passive best queue percentile；
* opposite best queue percentile；
* depth slope；
* microprice minus mid；
* 固定价格坐标下的 top-5 depth。

### 订单流状态

* OFI：10、50、100、500ms；
* signed trade volume：10、50、100、500ms；
* aggressive buy/sell intensity；
* previous same-side burst count；
* cancellation/addition flow；
* event arrival intensity。

### 市场环境

* realized volatility：100ms、1s、10s；
* time since last mid-price move；
* spread regime；
* local feed latency；
* funding phase、OI、liquidation 只作为 tags。

OFI 值得作为核心 pre-state，而不是只看成交量，因为短期价格变化与 best-level order-flow imbalance 的关系通常比单独 trade volume 更稳定；queue-reactive 模型则提供了一个“订单流强度取决于当前盘口状态”的可解释条件基线。([arXiv][4])

---

## Trigger (T)

保留：

* (Q_0)；
* (V_{p0})；
* (I_{\text{best}})；
* (I_{\text{burst}})；
* burst duration；
* number of prints；
* levels swept；
* threshold-crossing trade；
* local/exchange clock difference。

---

## Response (R)

同时保存两个坐标系统：

### 固定价格坐标

相对于 trigger-time (p_0) 和 mid：

[
\ell^{fixed}
============

\frac{p-p_0}{tick}
]

用来识别：

* 原价 refill；
* 撤退；
* reprice；
* 同价 liquidity persistence。

### moving-best 坐标

相对于当前 best：

[
\ell^{moving}
=============

\frac{p-p^{best}_t}{tick}
]

用来描述 contemporaneous book shape。

同时保存：

* 前 50、后 100 个事件；
* 真实时间网格；
* 推荐初始 horizons：

```text
1ms, 5ms, 10ms, 25ms, 50ms,
100ms, 250ms, 500ms, 1s
```

但只有 M1 确认可观察的 horizon 才进入正式结论。

此前的设计也强调，episode 必须同时保留 event-time response shape 与真实微秒 latency，并对价格、买卖方向和流动性尺度进行标准化。

---

## Outcome (O)

第一版定义 signed markout：

[
M_h
===

s\cdot
\frac{
mid_{t_0+h}-mid_{t_0}
}{
tick
}
]

其中：

* buy shock：(s=+1)；
* sell shock：(s=-1)。

因此：

* (M_h>0)：沿冲击方向继续移动，对 passive maker 不利；
* (M_h<0)：冲击反转。

保留：

* mid markout；
* microprice markout；
* best-price move；
* spread path；
* 后续 aggressive flow；
* 是否再次发生 qualifying shock。

先不要把公共 L2 上的“价格碰到报价”叫作 fill。真实 fill 需要 queue model 或自己的订单回报。

---

# M5：先做事件研究，不要立刻做 motif

这一阶段的目标是发现第一个可信的结构事实。

我建议把 trigger 后的市场响应理解成一个 **competing-risks race**：

[
T_R = \text{首次 aggregate replenishment}
]

[
T_P = \text{首次同方向价格移动}
]

[
T_W = \text{passive side 明显撤退或 reprice}
]

真正有用的问题是：

[
P(T_P<T_R\mid S^{pre},T)
]

也就是：

> 在当前状态和冲击程度下，价格先延续，还是流动性先恢复？

这比一开始聚类一堆深度矩阵更接近 maker 决策。

---

## 聚合 replenishment 的定义

因为只有 MBP，不要称为“同一 quote refill”。

可以定义 mechanical residual queue：

[
Q_t^{mechanical}
================

\max(Q_0-V_{p0,t},0)
]

观察到的 excess displayed depth：

[
X_t
===

Q_t(p_0)-Q_t^{mechanical}
]

其中：

* (X_t>0)：存在净新增/补充；
* (X_t<0)：存在额外撤单、数据时序差异或更深层消耗；
* 它仍然是 aggregate proxy，不是逐订单重建。

还可以定义：

[
RR_h
====

\frac{
\max(Q_h(p_0)-Q^{mechanical}*h,0)
}{
\min(V*{p0,h},Q_0)+\epsilon
}
]

这表示相对已消耗数量的 aggregate replenishment ratio。

---

## 第一版必须输出的曲线

### 剂量响应

按：

```text
[0.10, 0.25)
[0.25, 0.50)
[0.50, 0.75)
[0.75, 1.00)
[1.00, +∞)
```

比较：

* (P(T_R<h))；
* (P(T_P<h))；
* (P(T_W<h))；
* best depletion；
* same-price recovery；
* signed markout。

### 状态条件化

至少按以下变量分层：

* spread；
* (Q_0) percentile；
* L1 imbalance；
* OFI；
* short volatility；
* recent trade intensity。

### 对照组

不要用随机普通时刻作为主要 control。

使用：

> 相同 pre-state 下，impact ratio 较低的 aggressive trade packets。

这样对照组仍然经历了相同类型的机制事件，只是剂量较低。

### 跨日稳定性

每天单独计算 effect，再做 day-block bootstrap。不要把数百万 episode 当成数百万独立样本。

---

## M5 的候选“结构事实”

理想输出不是：

> 大单以后价格会涨。

而是类似：

> 在 1-tick spread、pre-trigger OFI 处于中性区间、(Q_0) 位于中等深度分位时，(I_{\text{best}}\ge0.5) 后 50ms 内没有出现 aggregate replenishment 的 episode，其 250ms 同方向 mid-price continuation 概率在多数交易日更高。

这才是能够继续支撑 maker action 的条件事实。

---

# M6：条件响应模型与 response taxonomy

只有 M5 显示存在稳定异质性后，才进入 response state。

第一版不要上 HMM。先建立一个可解释 taxonomy：

| Response state           | 定义方向                             |
| ------------------------ | -------------------------------- |
| `SAME_PRICE_ABSORPTION`  | 原价深度快速恢复，best 不消失或很快恢复，价格不延续     |
| `DEPLETION_CONTINUATION` | passive best 消失，mid 沿冲击方向移动      |
| `PASSIVE_RETREAT`        | passive side 多档深度同时减少或向外 reprice |
| `SPREAD_RECOVERY`        | spread 短暂扩大后快速回到原状态              |
| `MIXED/UNRESOLVED`       | 多事件重叠或观察窗口不足                     |

阈值不要在 final test 上调整。使用：

* discovery days：探索定义；
* calibration days：冻结阈值和模型；
* final test：只运行一次。

---

## 条件 baseline

模型目标是：

[
P(R\mid S^{pre},T)
]

而不是直接预测 PnL。

第一版可以采用：

* 分层概率表；
* regularized logistic regression；
* discrete-time hazard model；
* monotonic GAM；
* competing-risk model。

输入只使用 pre-state、trigger 和 decision-time 前已经看到的 response prefix。

如果在控制这些变量后仍存在重复 response residual，再考虑 PCA、kNN 或 motif graph。

也就是说：

> Motif 是第二层工具；
> competing risks 和条件响应概率才是第一层骨架。

---

# M7：冻结后，在 final test 上验证

在最后 15 天中：

1. 不重新选择 (\theta)；
2. 不调整 refill threshold；
3. 不改变 state taxonomy；
4. 不调整 feature list；
5. 不重新选择 horizon；
6. 不因 PnL 不好看而删 episode。

需要验证：

### Response reproducibility

* 每个 state 的出现频率；
* response prototype；
* refill/depletion hazard；
* 跨日覆盖；
* 买卖方向镜像稳定性。

### Outcome separation

taxonomy 冻结后再比较：

[
E[M_h\mid Z=k,S,T]
]

以及 state-matched controls。

### 建议的 go/no-go 标准

至少有一个 response state：

* 在 discovery、calibration、final test 中定义一致；
* 核心效应在大多数 test days 上方向一致；
* day-block bootstrap confidence interval 不包含零；
* 相对于只使用 (S^{pre},T) 的 baseline，有 out-of-sample predictive lift；
* 不是由单一 liquidation day 或极端行情日贡献。

如果不通过，这条 slice 的结论就是：

> 这个 anchor 在当前 venue/horizon 上没有足够稳定、可用于 maker policy 的条件响应结构。

这是一个合格的负结果，不应该硬聚类出五个“策略族”。

---

# M8：构建在线 prefix detector

完整 episode 只能事后看到，但 maker 需要早期判断。

定义：

[
\widehat Z_{\tau_d}
===================

f
\left(
S^{pre},
T,
R_{0:\tau_d}
\right)
]

其中 (\tau_d) 可以是：

```text
first 1 relevant event
first 3 relevant events
5ms
10ms
25ms
50ms
```

实际使用哪些，由 M1 的 feed cadence 决定。

---

## prefix features

* trigger 后是否立即出现同价正向 depth update；
* 当前 (Q(p_0))；
* book_ticker 上 p0 是否仍为 best；
* spread 是否扩大；
* continued aggressive volume；
* passive-side L1–L5 cancel flow；
* opposite-side depth response；
* microprice movement；
* elapsed real time；
* number of events since trigger。

## 输出不要只是分类标签

输出 calibrated probabilities：

[
p_{\text{cont}}
===============

P(T_P<T_R\mid\mathcal I_{\tau_d})
]

[
p_{\text{absorb}}
=================

P(T_R<T_P\mid\mathcal I_{\tau_d})
]

对 maker 来说，概率比硬分类更有用。

## 验收标准

* 只能使用本地已收到的数据；
* decision timestamp 明确；
* 加上计算延迟与 order entry latency 后，决策仍发生在主要价格结果前；
* calibration 稳定，而不是只有高 AUC；
* 在 final test 上保持效果；
* 同一事件重复 replay 得到完全相同结果。

---

# M9：接入 GLFT，但第一版只做 shock overlay

不要一开始为不同 state 重估整套 GLFT。

先保持原来的 GLFT baseline quote：

[
\delta^{GLFT}*{bid},
\qquad
\delta^{GLFT}*{ask}
]

再增加短生命周期 overlay。

对于主动买入冲击，被动危险一侧是 ask：

| 条件                    | 初始 maker action           |
| --------------------- | ------------------------- |
| (p_{\text{cont}}) 高   | cancel ask，进入短暂 pause     |
| (p_{\text{absorb}}) 高 | keep ask 或正常重新报价          |
| 不确定                   | ask size 降低或 widen 1 tick |
| opposite side 同时撤退    | 双边降 size，避免价格跳跃风险         |

可以写成：

[
\delta^{ask}
============

\delta^{GLFT}*{ask}
+
\Delta^{shock}*{ask}
]

[
q^{ask}
========

q^{GLFT}*{ask}
\cdot
m^{shock}*{ask}
]

其中第一版只允许：

```text
Δshock ∈ {0, 1 tick, cancel}
mshock ∈ {1.0, 0.5, 0}
```

这样 attribution 很清楚：收益变化究竟来自 shock overlay，还是来自整个 GLFT 被重新调了一遍。

---

## hftbacktest 评估

公共 MBP 无法给出真实 queue position，所以至少跑：

* RiskAverseQueueModel；
* 一种或多种 ProbQueueModel；
* PartialFillExchange；
* 多档 order-entry latency；
* 多档 order-response latency。

HftBacktest 明确区分 partial/no-partial fill exchange，并要求在 MBP 数据下使用 queue position model 推断排队位置；不同 queue model 会显著影响 fill simulation，因此最终模型必须用真实 live fills 校准。([HftBacktest][5])

建议 latency scenario：

```text
1ms
2ms
5ms
10ms
measured-live-distribution
```

在尚无自己的真实 order latency 分布前，不应只使用一个“乐观的 1ms”结果。

---

## Maker 指标

不要只看总 PnL：

### 执行

* fill rate；
* queue wait time；
* cancellation rate；
* orders per fill；
* time in market。

### 收益质量

* realized spread；
* post-fill markout：10/50/250ms/1s；
* adverse-fill ratio；
* maker rebate / fee；
* net PnL。

### 风险

* inventory variance；
* max inventory；
* shock-period drawdown；
* worst-decile episode PnL；
* liquidation-overlap PnL。

### 策略价值

对每个 decision state 计算：

[
EV(a\mid\mathcal I)
===================

P(F\mid a,\mathcal I)
\left[
\text{spread}
+\text{rebate}
--------------

E(\text{markout}\mid F,a,\mathcal I)
\right]
-------

## C_{\text{inventory}}

C_{\text{latency}}
]

---

## M9 的 go/no-go

shock overlay 必须同时满足：

1. final test 净结果不低于 baseline；
2. adverse-fill markout 改善；
3. worst-decile shock loss 不恶化；
4. 在至少两种 queue model 下方向一致；
5. 在多档 latency 下不是只在最乐观场景有效；
6. 效果不是来自极端降低 fill rate、几乎不挂单。

否则，它可能是一个不错的价格 predictor，但不是一个好的 market-making policy。

---

# M10：backtest/live alignment

核心路径应当独立于框架：

```text
Tardis CSV
    ↓
CanonicalEvent
    ↓
LocalBookState
    ↓
AggressiveQueueShockDetector
    ↓
PrefixResponseState
    ↓
ShockOverlayPolicy
    ↓
OrderCommand
```

然后分别接：

```text
HftBacktestAdapter
ReplayAdapter
PaperAdapter
LiveVenueAdapter
```

不要在 notebook 里实现一份 detector，再在 hftbacktest 中实现第二份，然后 live 中实现第三份。那种三胞胎通常长大后互不相认。

每次决策保存：

```text
trigger_id
decision_id
book_state_hash
feature_version
feature_values
model_version
action
order_command_id
venue_order_id
local_decision_timestamp
local_order_send_timestamp
exchange_ack_timestamp
```

你的 tick2order 目标可以直接定义为：

[
L_{\text{tick2order}}
=====================

## t_{\text{order send}}

t_{\text{trigger arrival}}
]

并同时监控：

* P50；
* P90；
* P99；
* GC / scheduling spikes；
* book-to-decision；
* decision-to-command；
* command-to-send。

## alignment 验收

把 live/paper 捕获的原始事件重新 replay：

* feature hash 应一致；
* trigger 应一致；
* decision 应一致；
* order command 应一致；
* 差异必须能够被明确归因于配置、数据缺失或并发时序。

同一份输入无法产生同一决策时，不要先调策略参数，先修 pipeline。

---

# 五、多交易所和 ETH 应该怎样加入

严格采用“一次只改变一个维度”。

## 第一层：同资产、换 venue

冻结 BTC primary venue 上的：

* trigger；
* normalization；
* feature definition；
* response taxonomy；
* horizon。

然后在第二 venue 上重新估计参数，但不改概念定义。

回答：

> 这是 BTC 市场普遍存在的 response structure，还是 primary venue 特有的撮合/参与者结构？

## 第二层：同 venue、换 ETH

在 primary venue 上运行 ETH。

回答：

> 这是同一 venue 的制度结构，还是 BTC 特有的流动性结构？

## 第三层：加入 cross-venue context

只有前两步完成后，才加入：

* 其他 venue 是否提前出现 aggressive shock；
* cross-venue mid move；
* leader venue OFI；
* basis dislocation；
* cross-venue liquidation。

Tardis 说明，同一采集服务器位置的 exchange local timestamps 可以直接比较，但跨地区采集的 local timestamps 不能未经校正就解释为严格先后关系。因此多 venue lead-lag 前必须检查 collection location、时钟偏移和 latency distribution。([Tardis Documentation][6])

顺序应是：

[
\text{单 venue 机制}
\rightarrow
\text{跨 venue 复现}
\rightarrow
\text{跨 venue 状态变量}
]

而不是一开始就把八个交易所拼进一个 embedding，让模型自己“悟道”。

---

# 六、建议的实际编码任务顺序

第一批只做下面 8 个任务，完成后才进入统计模型：

1. `research_contract.md`
   冻结 trigger、时钟、burst、阈值和 exclusion rules。

2. `tardis_audit.py`
   输出候选 venue 的 cadence、数据完整性和 event-count report。

3. `canonical_event.py`
   将 depth、trade、book_ticker 转成统一事件。

4. `l2_reconstructor.py`
   实现 snapshot epoch、绝对 amount 更新、top-25 view。

5. `book_validator.py`
   与 `book_snapshot_25`、`quotes`、`book_ticker` 对照。

6. `trade_book_merger.py`
   实现 local-time replay、exchange-time diagnostic 和 tie ambiguity。

7. `aggressive_queue_shock_detector.py`
   输出固定 schema 的 `ShockEvent`。

8. `episode_viewer.py`
   对任意 episode 绘制盘口、成交、impact 和 response 图。

第二批：

9. `episode_builder.py`
10. `pre_state_features.py`
11. `response_features.py`
12. `event_study.py`
13. `competing_risks.py`

第三批：

14. `response_taxonomy.py`
15. `prefix_detector.py`
16. `shock_overlay_policy.py`
17. `hftbacktest_adapter.py`
18. `decision_parity.py`

---

# 七、这条 slice 真正完成时应当得到什么

最终交付不是一句：

> “大单吃队列以后可能会涨。”

而应该是一套可运行的结论：

[
\boxed{
\begin{aligned}
&\text{在什么 pre-state 下发生了 clean aggressive queue shock}\
&\rightarrow
\text{市场更可能先 refill、retreat 还是 price continuation}\
&\rightarrow
\text{在第几个事件或多少毫秒时能够可靠判断}\
&\rightarrow
\text{我的 maker quote 应 keep、cancel 还是 widen}\
&\rightarrow
\text{计入 queue、latency、fees 和 adverse fill 后是否提高 EV}
\end{aligned}
}
]

其中最关键的五个验收节点是：

1. **数据可信**：盘口能够被精确重建；
2. **事件可信**：任意 trigger 都可解释和复现；
3. **结构可信**：条件响应跨日存在；
4. **信号可执行**：在结果发生前可以在线识别；
5. **策略有价值**：在现实 queue/latency 假设下改善 maker EV。

眼下应直接推进到 **M3 trigger detector 验收**。在这之前，不做 motif，不调 GLFT，不看策略 PnL。因为这条 slice 的地基不是模型，而是你是否真的知道：**哪一笔主动成交，吃掉了哪一个当时可见的队列。**

[1]: https://docs.tardis.dev/downloadable-csv-data-types?utm_source=chatgpt.com "Data Types | Tardis.dev Documentation"
[2]: https://docs.tardis.dev/downloadable-csv-files?utm_source=chatgpt.com "Downloadable CSV files | Tardis.dev Documentation"
[3]: https://hftbacktest.readthedocs.io/en/latest/reference/hftbacktest.data.utils.tardis.html?utm_source=chatgpt.com "hftbacktest.data.utils.tardis module - Read the Docs"
[4]: https://arxiv.org/abs/1011.6402?utm_source=chatgpt.com "The Price Impact of Order Book Events"
[5]: https://hftbacktest.readthedocs.io/en/py-v2.1.0/order_fill.html?utm_source=chatgpt.com "Order Fill — hftbacktest - Read the Docs"
[6]: https://docs.tardis.dev/faq/data?utm_source=chatgpt.com "Data"
