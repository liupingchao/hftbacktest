可以，而且这个方向更接近你真正要优化的对象：**不是交易所抽象意义上的市场，而是你的交易服务器实际能观察、计算并采取行动的市场。**

但我会把“跳过 Tardis 的研究阶段”改成：

> **把研究阶段搬进生产数据路径：实时采集、确定性重放、影子决策、小资金随机化实验、版本化迭代。**

Tardis 数据不是“虚假数据”，它只是 **Tardis 采集节点所看到的市场**，不等于你的服务器看到的市场。相反，你自己的 live feed 也不是撮合引擎的绝对真相：公共 L2 仍然存在聚合、批处理、隐藏流动性和传输延迟。即使自行采集，普通 L2 仍然只能看到聚合价位响应，不能识别是否来自同一个 maker。

因此第一条 vertical slice 应当定义为：

[
\boxed{
\text{真实生产 feed}
\rightarrow
\text{queue shock 候选}
\rightarrow
\text{queue shock 确认}
\rightarrow
\text{判断另一 venue 是否尚未反应}
\rightarrow
\text{KEEP / CANCEL 危险侧报价}
\rightarrow
\text{真实订单回报}
\rightarrow
\text{事件级 EV 评估}
}
]

---

# 一、先把第一条 slice 收紧成一个明确问题

先选 BTC 永续：

* Binance：BTCUSDT USDⓈ-M perpetual；
* Hyperliquid：BTC perpetual；
* 两边都实现行情采集、盘口和 shock detector；
* 先不开启双向跨所策略；
* 用 shadow 数据判断究竟是：

  * Binance → Hyperliquid 更有可执行 lead；
  * Hyperliquid → Binance 更有可执行 lead；
  * 还是两个方向都不够快，只能做 venue-local risk overlay。

第一版 maker action 只有两个：

[
A\in
{
\text{KEEP},
\text{CANCEL_RISK_SIDE}
}
]

不要同时引入：

* widen；
* resize；
* reprice；
* inventory skew 重估；
* GLFT 参数切换；
* 跨所 hedge。

第一版只回答：

> 当一个 venue 出现可解释的 aggressive queue shock 时，取消另一个 venue 的危险侧 maker quote，是否能减少 adverse fills，并且减少的损失大于放弃的 spread capture？

例如 Binance 主动买入 shock：

* 危险侧是 Hyperliquid ask；
* control：保持 ask；
* treatment：取消 ask；
* bid 不动；
* 到恢复条件满足后重新报价。

这会让策略效果非常容易归因。

---

# 二、最重要的原则：分子和分母必须来自同一个 venue

对 venue (v)，定义：

[
I_v
===

\frac{
\sum_{j\in B_v}
q_j
\mathbf 1(p_j=p_{0,v})
}{
Q_{0,v}
}
]

其中：

* (p_{0,v})：该 venue 上 trigger 前的 passive best price；
* (Q_{0,v})：该 venue 上 trigger 前的 passive best displayed queue；
* (B_v)：该 venue 上的一段同方向 aggressive trade burst。

绝对不要定义：

[
\frac{
V_{\mathrm{Binance}}
}{
Q_{\mathrm{Hyperliquid}}
}
]

它没有稳定的微观结构意义，因为两个 venue 的：

* feed batching；
* tick size；
* displayed liquidity；
* hidden liquidity；
* matching cadence；
* 合约参与者；

都不同。

跨 venue 的正确连接方式是：

[
I_{\mathrm{Binance}}
+
S_{\mathrm{Hyperliquid}}
\rightarrow
A_{\mathrm{Hyperliquid}}
]

即：

* shock severity 来自 source venue；
* target venue 的盘口状态独立作为 context；
* maker action 发生在 target venue。

---

# 三、当前两个交易所的行情现实，会直接改变 detector 设计

## 0. Historical SKHYNIX archive correction (2026-08-14)

The SKHYNIX datasets currently available in this repository do not implement
the idealized Binance feed contract described below. Their frozen collector
uses:

```text
wss://fstream.binance.com/ws

SKHYNIXUSDT@trade
SKHYNIXUSDT@depth@0ms
SKHYNIXUSDT@bookTicker
```

The archived `trade` payloads contain quantity `q` but no `nq` field or
equivalent RPI participation flag. Historical research therefore cannot
reconstruct an RPI-adjusted numerator. For those datasets:

```text
numerator semantics:
  observed @trade q

denominator semantics:
  visible pre-trigger queue

RPI adjustment:
  unavailable from historical archive

depth stream:
  @depth@0ms, with cadence measured from actual receipt data
```

The historical ratio is an observed trade-pressure proxy, not exact visible
queue consumption. The `aggTrade.nq` and `diff depth @100ms` text below is a
future target production-feed design; it must not be cited as the collection
contract of the existing SKHYNIX campaigns.

## 1. Binance target production feed design

截至 2026 年 8 月，Binance USDⓈ-M Futures 已经拆分为 public、market 和 private WebSocket 路径。`bookTicker` 是实时 BBO 更新，diff depth 最快是 100ms，而 futures `aggTrade` 会将相同价格和 taker side 的成交按 100ms 聚合。官方也建议不同类型的流使用分离连接，降低单连接负载和抖动。([Binance 开发者中心][1])

因此 Binance 上：

[
\frac{\text{aggTrade volume}}{\text{pre-trigger best queue}}
]

不能被解释成“精确逐笔消耗比例”，它更准确地是：

> **交易服务器观察到的 100ms aggressive microbatch，相对于此前可见 best queue 的压力比例。**

建议接入三条独立连接：

```text
Public:
  BTCUSDT bookTicker
  BTCUSDT diff depth @100ms

Market:
  BTCUSDT aggTrade

Private:
  ORDER_TRADE_UPDATE
  ACCOUNT_UPDATE
```

Binance 本地 L2 必须按 snapshot 加 diff depth 重建，并验证 `U/u/pu` 连续性；数量是更新后的绝对数量，`pu != previous u` 时要重新初始化。([Binance 开发者中心][2])

### 一个非常关键的 Binance 细节

Binance 当前 `aggTrade.q` 包括涉及 RPI orders 的成交，而 RPI liquidity 不出现在公共 depth 和 bookTicker 中；`nq` 表示不涉及 RPI 的 normal quantity。因此这个 slice 的 numerator 应优先使用：

[
V_{\mathrm{visible}}
====================

\texttt{nq}
]

而不是：

[
V_{\mathrm{all}}
================

\texttt{q}
]

否则分子可能包含对不可见 RPI liquidity 的成交，分母却只有可见队列，ratio 会系统性失真。([Binance 开发者中心][3])

---

## 2. Hyperliquid：低延迟研究不要只依赖公共 `l2Book`

Hyperliquid 的公共接口可以订阅 `l2Book`、`bbo` 和 `trades`；官方 SDK 中，L2 是聚合价位结构 `{px, sz, n}`，trade 包含 coin、side、price、size、hash 和 time 等字段。它仍然是聚合 book，不是逐订单 diff。([GitHub][4])

对于真正依赖 queue shock、refill 和撤退速度的策略，建议把 Hyperliquid 的生产数据平面升级为 non-validating node：

```text
--write-trades
--write-raw-book-diffs
--write-order-statuses
--stream-with-block-info
--disable-output-file-buffering
```

官方 node 项目支持记录 trades、raw book diffs 和 order statuses，并可以携带 block time、block number 和 local time。官方当前还列出了较高的主机资源需求和很大的日志量，所以不应该毫无隔离地把 node、研究写盘和低延迟 order router 塞进同一个 CPU/磁盘资源池。([GitHub][5])

更稳妥的部署是：

```text
同一地区 / 同一机房

Hyperliquid node process
    ↓ shared memory / unix socket
Strategy market-data adapter
    ↓
Strategy core
```

如果必须同一台机器：

* node、collector、strategy 分别 pin CPU；
* strategy hot path 不直接写磁盘；
* node 日志放独立 NVMe；
* raw recorder 走异步 ring buffer；
* strategy ingress 处重新打本地时间戳。

对于策略而言，**真正重要的时间不是 node 写文件的时间，而是 normalized event 到达策略进程的时间。**

公共 Hyperliquid WebSocket 可以保留，用于：

* parity check；
* reconnect bootstrap；
* 外部健康检查；
* 发现 node 数据异常；

但不要把它当低延迟 L2 queue truth。

---

# 四、跨交易所对齐，不是把 exchange timestamp 强行排成一条时间线

你真正需要的是两套时间：

## 1. Strategy-observed time

[
t_{\mathrm{obs}}
]

表示消息什么时候真正进入你的可决策信息集。

建议至少记录：

```text
t_socket_read_mono
t_decode_done_mono
t_strategy_publish_mono
```

主排序使用同一台机器上的 monotonic clock，例如 `CLOCK_MONOTONIC_RAW`。

如果两个 feed 最终进入同一策略进程：

> Binance 和 Hyperliquid 的跨 venue 因果顺序，应以本地 `t_strategy_publish` 排序。

## 2. Venue semantic time

同时保留：

```text
Binance E
Binance T
Binance U/u/pu

Hyperliquid block_time
Hyperliquid block_number
Hyperliquid event time
trade id / hash
```

它们用于：

* 检查 feed latency；
* 判断同 venue 消息语义；
* 诊断网络抖动；
* 重建 block 或 update sequence。

但不要为了让图看起来更整齐，按 exchange timestamp 重排 live strategy 已经看到的事件。

否则会出现一种危险的伪回测：

> 回放时知道交易所内部哪个事件先发生，但 live 当时消息还没到服务器。

---

# 五、第一条 slice 需要“两阶段 shock detector”

这点非常重要。

因为 Binance `aggTrade` 是 100ms 聚合，如果等到 confirmed ratio 到达才取消报价，可能已经太晚。

所以必须区分：

## Stage A：早期候选 `ShockCandidate`

基于实时 bookTicker 或 node book diff：

[
C_t
===

\frac{
(Q_0-Q_t)^+
}{
Q_0
}
]

候选条件可以是：

* 同一个 best price 的显示队列突然下降；
* passive best 完全消失；
* source venue BBO 向冲击方向移动；
* 已到达的部分 aggressive trades 已经达到较低阈值。

这个事件很快，但不纯净：

* 队列下降可能来自成交；
* 也可能来自撤单；
* 还可能是 feed batching。

因此它只能叫 candidate，不能叫 confirmed aggressive shock。

## Stage B：后续确认 `ShockConfirmed`

当 trade feed 到达后：

[
I_v
===

\frac{
V_{p_0,v}
}{
Q_{0,v}
}
\ge \theta
]

满足时确认。

对 Binance：

[
V_{p_0,B}
=========

\texttt{aggTrade.nq}
]

对 Hyperliquid node：

* 使用该 block 或事件序列中的 aggressive trades；
* 只累计在 (p_0) 上发生的成交；
* 向更深价位 sweep 的部分另存，不进入第一版主 ratio。

因此研究标签仍然是：

[
I_v\ge\theta
]

但在线策略使用的是：

[
P(
I_v\ge\theta
\mid
\text{当前已观察到的 candidate prefix}
)
]

也就是说：

> confirmed shock 是事后可信标签；
> candidate shock 是实时可执行信号。

这正是该 slice 从“研究现象”走向“maker 对策”的桥。

---

# 六、建议的 trigger event contract

第一版主阈值可以保留：

[
\theta=0.5
]

但从第一天开始同时记录连续 ratio，并生成：

```text
[0.25, 0.50)
[0.50, 0.75)
[0.75, 1.00)
[1.00, +∞)
```

不要为了近期 PnL 不断改 detector 阈值。

每个 shock 至少保存：

```text
shock_id
source_venue
instrument
direction
measurement_model
  # binance_100ms_microbatch
  # hyperliquid_node_block
candidate_time_obs
confirmed_time_obs
p0
q0
q0_age
aggressive_volume_at_p0
total_aggressive_volume
impact_ratio
book_version
connection_epoch
sequence_or_block
candidate_reason
confirmation_status
quality_class
```

## Clean shock 的条件

```text
book valid
无 sequence gap
无 reconnect/reset
q0 > 0
trade price 与 p0 一致
book age 在可接受范围内
source 与 target feed 都健康
没有前一个未结束 shock
```

## 三档质量

```text
A: clean，可用于 live action 和正式统计
B: batching / ordering 有歧义，只用于研究
C: gap / stale / reconnect，完全排除
```

不要删除 B 类原始数据；保存它们有助于判断 feed ambiguity 是否本身与行情剧烈程度相关。

---

# 七、跨 venue 最重要的研究对象不是 correlation，而是 actionability margin

假设 source venue 在 (t_s) 产生 shock candidate，target venue 在 (t_r) 发生危险结果：

* own ask fill；
* target mid 向上移动；
* target best ask 被耗尽；
* target local shock 出现。

你的实际取消生效时间是：

[
t_{\mathrm{cancel-effective}}
=============================

t_s
+
L_{\mathrm{decode}}
+
L_{\mathrm{decision}}
+
L_{\mathrm{send}}
+
L_{\mathrm{venue}}
]

定义：

[
M
=

## t_r

t_{\mathrm{cancel-effective}}
]

其中：

* (M>0)：理论上来得及；
* (M<0)：市场已经先反应，signal 再准也不可执行。

需要分别估计：

[
P(M>0\mid I_B,S_H)
]

和：

[
P(M>0\mid I_H,S_B)
]

这一步会决定最终使用：

```text
Binance → Hyperliquid
Hyperliquid → Binance
仅 Hyperliquid local
仅 Binance local
```

不要预设 Binance 一定 lead Hyperliquid。对你的策略而言，lead 由：

* 市场真实的信息传播；
* 交易所 feed batching；
* 网络路径；
* server region；
* decoder；
* process scheduling；

共同决定。

这里“谁 lead 谁”不是宇宙真理，而是你当前生产拓扑的可交易事实。

---

# 八、顺序可执行的里程碑

| 节点 | 目标                             | 交付物                                | 通过条件                                             |
| -- | ------------------------------ | ---------------------------------- | ------------------------------------------------ |
| M0 | 冻结研究和 action contract          | `slice_contract_v1`                | ratio、candidate、confirmation、action、outcome 均无歧义 |
| M1 | 得到可重放的真实服务器 view               | raw journal + replay               | 相同 journal 每次产生完全相同的 canonical events            |
| M2 | 建立两个 venue 的可信 book            | Binance L2 + Hyperliquid node book | 无未解释 sequence/block divergence                   |
| M3 | 对齐 private order feedback      | own order state machine            | send、ack、resting、partial fill、fill、cancel 全链路闭合  |
| M4 | 验证 shock detector              | `shock_events`                     | 随机抽取任意事件都能解释 (p_0,Q_0,V,\theta)                  |
| M5 | 建立跨 venue lead/actionability 图 | lead-lag report                    | 找到至少一个方向存在正的 actionability margin                |
| M6 | 运行 shadow policy               | shadow decisions                   | 候选 policy 在危险 fill 前有足够提前量                       |
| M7 | 小风险随机化 canary                  | control/treatment fills            | treatment 改善 event-level EV，不只是减少 fill           |
| M8 | 版本化迭代                          | champion/challenger registry       | 每次只改变一个假设或参数族                                    |
| M9 | 扩展 action space                | resize/widen/reprice               | 在 KEEP/CANCEL 基线之后才增加复杂动作                        |

---

# 九、每个节点应该具体完成什么

## M0：冻结 slice contract

建议第一版固定：

```text
asset                   = BTC perpetual
primary_label_theta     = 0.50
severity_bands          = [0.25, 0.50, 0.75, 1.00]
action_space            = KEEP / CANCEL_RISK_SIDE
one_action_per_shock    = true
cooldown_overlap        = disabled
outcome_in_detector     = false
cross_venue_clock       = local monotonic arrival
```

Primary hypothesis：

[
I_{\mathrm{source}}\uparrow
\Rightarrow
P(\text{target adverse fill})\uparrow
]

Policy hypothesis：

[
\text{Cancel dangerous quote after actionable candidate}
\Rightarrow
\Delta EV>0
]

其中：

[
\Delta EV
=========

## \text{avoided adverse markout}

## \text{missed spread capture}

\text{extra cancel/requote costs}
]

---

## M1：raw journal 与 deterministic replay

Hot path：

```text
socket
  → timestamp
  → raw payload journal ring
  → decoder
  → canonical event
  → strategy
```

Recorder 不得阻塞 strategy。

每条 raw record 保存：

```text
venue
connection_id
connection_epoch
rx_sequence
local_monotonic_time
local_wall_time
raw_payload
payload_crc
software_version
```

然后提供两种 replay：

```text
original-time replay
accelerated deterministic replay
```

回放时必须使用与 live 完全相同的：

* decoder；
* book builder；
* shock detector；
* strategy policy。

不要另外写一个“研究版 detector”。那个双胞胎最后大概率会翻脸。

---

## M2：盘口和 feed health

### Binance

维护：

```text
fast_bbo_view     <- bookTicker
consistent_l2     <- snapshot + diff depth
```

`fast_bbo_view` 用来获得较新 (Q_0)。

`consistent_l2` 用于：

* top-5/top-10 context；
* imbalance；
* OFI；
* depth slope；
* book reconstruction QA。

两个 view 都保存 update ID，并检查 divergence。

### Hyperliquid

维护：

```text
node_raw_book_diffs
  + bootstrap snapshot
  → local aggregated L2
```

公共 `l2Book` 作为 validation oracle。

### Health gate

只要出现：

```text
sequence gap
stale book
node lag
reconnect epoch
decoder backlog
clock anomaly
```

该 slice 禁止发 treatment action。

对于 maker 策略，最危险的模型不是错误模型，而是拿着 300ms 前的盘口一本正经地下单。

---

## M3：先闭合自己的订单状态

每笔订单需要：

```text
client_order_id
venue_order_id
strategy_version
shock_id
send_time
ack_time
resting_time
cancel_send_time
cancel_ack_time
partial_fill_times
fill_times
fill_prices
fees
```

Binance private user stream提供自己的订单更新，Hyperliquid SDK 也提供 `orderUpdates` 和 `userFills` 类型。([Binance 开发者中心][6])

这一步意义很大。只有公共 L2 时，你研究的是：

[
S\rightarrow R\rightarrow O
]

有自己的订单回报后，你终于可以研究：

[
S\rightarrow A\rightarrow O
]

包括真实 action、fills、inventory、realized spread 和 markout。

---

## M4：shock detector 验收

必须有一个 episode viewer，统一展示：

```text
source venue BBO/L2
source trades
source ratio path
target venue BBO/L2
own orders
candidate time
confirmation time
decision time
cancel send/ack
fills
mid markout
```

至少人工检查以下类别：

* buy / sell；
* (I\in[0.5,0.75))；
* (I\in[0.75,1))；
* (I\ge1)；
* candidate confirmed；
* candidate rejected；
* best survives；
* best disappears；
* sequence clean；
* batching ambiguous。

通过标准是：

> 任意拿一个 `shock_id`，都可以完整解释为什么它被触发，策略当时具体看到了什么。

---

## M5：shadow 研究跨所 actionability

对每个 source shock，构造 target response：

[
R=
(
T_{\mathrm{target\ trade}},
T_{\mathrm{target\ BBO\ move}},
T_{\mathrm{target\ queue\ depletion}},
T_{\mathrm{own\ fill}}
)
]

按以下条件分层：

* source impact ratio；
* source queue percentile；
* target spread；
* target imbalance；
* target OFI；
* source-target basis；
* own quote distance；
* own order age；
* inventory；
* time since target last move。

第一版模型用：

* 分箱；
* Kaplan–Meier / competing risks；
* logistic regression；
* 简单 hazard model。

先不要使用神经网络。

核心输出：

[
P(
\text{own adverse fill before cancel effective}
\mid
\text{candidate},S
)
]

以及：

[
P(
\text{confirmed shock}
\mid
\text{candidate prefix}
)
]

---

# 十、第一版 live policy

假设 shadow 结果显示 Binance → Hyperliquid 有足够 lead。

## Buy shock

Binance aggressive buy shock：

```text
Binance passive ask 被冲击
→ Hyperliquid ask 是危险侧
```

Treatment 条件：

```text
source candidate quality = A
source severity probability sufficiently high
Hyperliquid book healthy
own ask currently resting
own ask 距离 best ask <= 1 tick
inventory 不要求强制卖出
actionability margin positive
没有未结束 shock
```

Action：

```text
cancel Hyperliquid ask
keep Hyperliquid bid
do not immediately reprice ask
```

## Resume 条件

满足任一条件后恢复 baseline quote：

```text
target mid 已沿 shock 方向移动
target passive side 出现稳定 replenishment
出现 opposite shock
固定 cooldown 到期
inventory risk override
```

卖出 shock 完全镜像。

第一版不要在 shock 后：

* 主动追单；
* 立即跨价 hedge；
* 扩大另一侧 size；
* 预测方向后加 inventory。

先证明“避开坏 fill”本身有价值。

---

# 十一、live feedback 应采用随机化 canary，而不是看一两笔亏损就调参数

对于每个 qualifying shock：

```text
hash(shock_id) % 2
```

决定：

* control：保持 baseline；
* treatment：执行 cancel overlay。

这样：

* 分组可以 deterministic replay；
* 不需要运行时随机数；
* 不会因为市场状态手动挑 treatment；
* 可以直接比较同类 episode。

评估至少包括：

## 执行质量

* trigger-to-decision；
* decision-to-send；
* send-to-ack；
* cancel-send-to-cancel-ack；
* cancel race fills；
* orders per fill；
* cancel rate。

## Fill 质量

* fill probability；
* realized spread；
* 10/50/100/250/500ms/1s post-fill markout；
* adverse-fill ratio；
* missed favorable fills。

## 经济价值

[
EV_{\mathrm{event}}
===================

\text{spread capture}
+\text{rebate}
-\text{markout}
-\text{fees}
-\text{inventory cost}
]

control 与 treatment 的差异必须按：

* day block；
* volatility regime；
* buy/sell side；
* source venue；

分别报告。

不能把几万条同一小时的事件当成几万个独立样本。

---

# 十二、策略迭代不应是“每拿到一笔反馈就改”

建议把参数分成两类。

## 快速状态参数

可以每个 event 更新：

* 当前 inventory；
* book；
* latency；
* feed health；
* current shock state；
* current candidate probability。

## 慢速 policy 参数

只能版本化更新：

* (\theta)；
* candidate threshold；
* context filter；
* cooldown；
* resume rule；
* cancel vs size reduction；
* source-target direction。

正确循环是：

```text
v1 冻结
→ 收集 control/treatment
→ replay
→ event-level evaluation
→ 提出单一修改
→ v2 canary
```

而不是：

```text
今天亏了
→ θ 从 0.5 改 0.7
→ 明天少成交
→ 改回 0.4
→ 后天行情变了
→ 开始怀疑宇宙
```

每次只改一个参数族，否则无法知道改进来自哪里。

---

# 十三、建议的代码模块顺序

实际开工顺序可以是：

```text
01_clock/
  本地 monotonic timestamp 和 latency decomposition

02_journal/
  raw append-only journal + CRC + replay

03_binance_adapter/
  bookTicker / depth / aggTrade / private orders

04_hyperliquid_adapter/
  node trades / raw book diffs / order statuses / user fills

05_books/
  FastBBO + ConsistentL2 + health state

06_canonical_events/
  统一 trade / depth / bbo / own-order schema

07_shock_detector/
  Candidate / Confirmed / Rejected state machine

08_cross_venue_aligner/
  source shock + target state + actionability margin

09_policy/
  KEEP / CANCEL_RISK_SIDE

10_execution/
  order command + cancel race state machine

11_episode_store/
  完整 S-T-A-O episode

12_analysis/
  lead curves / markout / realized spread / day-block CI

13_canary/
  deterministic control-treatment assignment
```

---

# 十四、这条 slice 的最终验收标准

只有五个 gate 全部通过，才算真正完成：

## Gate 1：数据真实性

同一个 raw journal 可以确定性重放，book 和 shock event 完全一致。

## Gate 2：事件真实性

(Q_0)、aggressive volume、ratio 和 candidate/confirmation 顺序都能解释。

## Gate 3：可执行性

至少一个 source→target 方向存在正的：

[
t_{\mathrm{target\ adverse\ event}}
-----------------------------------

t_{\mathrm{cancel-effective}}
]

## Gate 4：经济相关性

shock 对 own adverse fills 有稳定区分度，而不只是预测 public mid。

## Gate 5：真实策略增益

随机化 live treatment 在计入：

* missed fills；
* fees；
* cancel/requote；
* inventory；

之后仍提高：

[
EV_{\mathrm{maker}}
]

---

## 最终方案的核心变化

你不是先建完一套“完美基建”，然后才开始研究策略。

第一条 vertical slice 本身就贯穿：

[
\boxed{
\text{采集}
\rightarrow
\text{对齐}
\rightarrow
\text{重放}
\rightarrow
\text{事件识别}
\rightarrow
\text{跨所状态}
\rightarrow
\text{maker action}
\rightarrow
\text{订单反馈}
\rightarrow
\text{版本迭代}
}
]

而且第一个真正应该实现的在线对象，不是最终的：

[
I_v\ge\theta
]

而是：

[
\boxed{
P(
I_v\ge\theta
\mid
\text{当前服务器已经看到的信息}
)
}
]

因为 confirmed shock 负责给研究提供可信标签，**early candidate 才负责给 maker 留下取消报价的时间。**

[1]: https://developers.binance.com/en/docs/products/derivatives-trading-usds-futures/websocket-market-streams/Important-WebSocket-Change-Notice?utm_source=chatgpt.com "Important WebSocket Change Notice — Base URL Split & ..."
[2]: https://developers.binance.com/en/docs/products/derivatives-trading-usds-futures/websocket-market-streams/How-to-manage-a-local-order-book-correctly "How to manage a local order book correctly | Binance Developer Docs"
[3]: https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/ws-streams/market "Market - Futures (USDⓈ-M) WebSocket Market Streams | Binance Developer Docs"
[4]: https://github.com/hyperliquid-dex/hyperliquid-python-sdk/blob/master/hyperliquid/utils/types.py "hyperliquid-python-sdk/hyperliquid/utils/types.py at master · hyperliquid-dex/hyperliquid-python-sdk · GitHub"
[5]: https://github.com/hyperliquid-dex/node?utm_source=chatgpt.com "hyperliquid-dex/node"
[6]: https://developers.binance.com/docs/derivatives/usds-margined-futures/user-data-streams?utm_source=chatgpt.com "User Data Streams Connect"
