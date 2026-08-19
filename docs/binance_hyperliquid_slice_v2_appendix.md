对，这一版已经足以成为整个 **episode research architecture 的理论核心**。它比“围绕 trigger 提取特征”更深一层：**trigger 实际上定义了一套随机时钟，把 calendar-time process 转换成可比较的 event-conditioned samples。**

原始材料中“围绕明确事件设置 (\tau=0)，同时保留事件序号上的响应形状和真实微秒延迟”，正好支持你现在这个定位。

不过，我会再做四个关键收紧。做完以后，这套定义就不仅理论正确，而且能直接约束 live collector、replay、研究和策略代码。

---

# 一、Trigger 应称为“可观测停止时刻”，不要称为“因果原点”

你写的：

> (\tau_{\text{candidate}}) 是最早的因果 trigger 原点。

这里“因果”略微过强。

主动成交冲击、队列耗尽、跨所价差扩大都是**内生市场事件**。它们可能和后续价格变化共同受到某个隐藏信息流驱动，并不是实验意义上的外生干预。

更准确的名称是：

> **最早可观测、可在线计算的 event-time anchor。**

定义交易服务器在时刻 (t) 已经获得的信息为：

[
\mathcal F_t^{obs}
]

那么 candidate time 必须是相对于这套信息流的 stopping time：

[
\tau_i^{cand}
=============

\inf
\left{
t:
C\left(\mathcal F_t^{obs}\right)=1
\right}
]

它必须满足：

> 在任意时刻 (t)，只用截至 (t) 已到达服务器的信息，就能判断 (\tau_i^{cand}\le t) 是否成立。

这条定义看似数学洁癖，实际上直接决定有没有 look-ahead。

例如：

* 不能用后来到达的 trade 消息回头修改 candidate time；
* 不能按 exchange timestamp 重排后，把一个 live 时没有看到的事件放到 trigger 前；
* 不能用 burst 最终总成交量决定 burst 开始时是否构成 candidate；
* 不能使用未来的 confirmation 结果筛掉当时“错误”的 candidate，再评估在线策略。

因此更准确的表述是：

[
\boxed{
\text{Trigger 是一个基于服务器可观测信息定义的 stopping-time anchor，}
}
]

而不是一个因果 intervention。

---

# 二、五个时间点还不够清楚，建议拆成六个

你目前定义：

```text
τ_start
τ_candidate
τ_confirm
τ_decision
τ_effective
```

这里最大的潜在歧义是：

* `candidate` 是快速但有噪声的盘口异常；
* 还是 aggressive volume ratio 第一次达到 (\theta)？

如果 `candidate` 已经表示：

[
\frac{V_{p_0}}{Q_0}\ge\theta
]

首次成立，那么 `confirm` 又确认什么，需要额外说明。

建议采用下面六个锚点：

[
\mathcal A_i=
\left(
\tau_i^{start},
\tau_i^{cand},
\tau_i^{\theta},
\tau_i^{confirm},
\tau_i^{decision},
\tau_i^{effective}
\right)
]

## 1. (\tau^{start})：burst 开始

第一笔可能属于此次冲击的 aggressive trade，或第一个异常 queue-depletion update 到达服务器。

## 2. (\tau^{cand})：快速候选信号出现

例如：

* passive best queue 快速下降；
* passive best 消失；
* 已观察到的部分成交量达到较低预警阈值；
* 跨所 gap 快速扩大。

这个时间最早，但允许 false positive。

## 3. (\tau^\theta)：主 trigger 阈值被观察到越过

[
\tau_i^\theta
=============

\inf
\left{
t\ge\tau_i^{cand}:
\widehat I_i(t)
===============

\frac{V_{p_0,i}(t)}{Q_{0,i}}
\ge\theta
\right}
]

这是 `aggressive volume / pre-trigger best queue` 真正第一次达到主阈值的时刻。

## 4. (\tau^{confirm})：语义与数据质量确认

例如确认：

* trade side、price 与 (p_0) 一致；
* 没有 sequence gap；
* book 没有 stale；
* 后续 feed reconciliation 没有否定该事件；
* 该事件属于 clean quality class。

它可以晚于阈值越过时间，但不能回头进入之前的策略信息集。

## 5. (\tau^{decision})：策略输出动作

[
A_i
===

\pi\left(\mathcal F_{\tau_i^{decision}}^{obs}\right)
]

## 6. (\tau^{effective})：动作在交易所生效

例如 cancel acknowledgement，或通过订单状态确认订单已不再 resting。

于是所有相对延迟都以 candidate 为基础记录：

[
d_i^\theta
==========

\tau_i^\theta-\tau_i^{cand}
]

[
d_i^{confirm}
=============

\tau_i^{confirm}-\tau_i^{cand}
]

[
d_i^{decision}
==============

\tau_i^{decision}-\tau_i^{cand}
]

[
d_i^{effective}
===============

\tau_i^{effective}-\tau_i^{cand}
]

这样可以同时研究：

* candidate 是否足够早；
* 阈值达到时是否已经太晚；
* confirmation 增加了多少精度，又牺牲了多少时效；
* cancel 是否在 adverse event 前真正生效。

---

# 三、不要只建立一种 episode，而应建立两个相互关联的 episode family

这是现在最重要的进一步细化。

## Family A：Candidate-aligned policy episodes

它包含**所有 candidate**：

* 后来 confirmed 的；
* 后来 rejected 的；
* 因数据中断无法确认的；
* 没有达到最终 (\theta) 的。

定义：

[
E_i^{policy}
============

\left(
\mathcal A_i,
S_i^{pre},
T_i^{0:d_i},
R_i^{0:d_i},
A_i,
R_i^{d_i:H},
O_i^{market},
O_i^{own},
C_i
\right)
]

其中 (C_i) 是数据质量和 censoring 状态。

它用于学习：

[
P(
\text{最终确认},
R^{future},
O
\mid
S^{pre},
T^{prefix},
R^{prefix}
)
]

以及不同动作的价值。

这个 family 必须保留 false positives，因为 live 策略在 candidate 时并不知道它未来会不会被确认。

如果只保留 confirmed cases，相当于用未来信息筛选了在线信号：

[
P(O\mid candidate,\ confirmed=1)
]

它不等于 live 时真正面对的：

[
P(O\mid candidate)
]

---

## Family B：Confirmed-shock research episodes

它只包含：

[
I_i\ge\theta
]

且通过数据质量确认的 clean shocks。

它用于研究：

* 冲击剂量和响应之间的关系；
* refill、撤退、target catch-up 的分布；
* 不同 (\theta) 下的 dose-response；
* 相似 (S^{pre},T) 下的条件响应结构。

可以写成：

[
E_i^{research}
==============

E_i^{policy}
\mid
\left(
confirmed_i=1,
quality_i=A
\right)
]

这两个 family 回答不同问题：

| Episode family         | 对齐原点                  | 回答的问题                 |
| ---------------------- | --------------------- | --------------------- |
| Candidate-aligned      | (\tau^{cand})         | live 时是否应该行动          |
| Confirmed-shock        | (\tau^\theta) 或机制事件时刻 | 可信冲击后市场如何响应           |
| Effective-aligned view | (\tau^{effective})    | cancel race 和执行结果如何发生 |

它们不应成为三套不同数据，而应是**同一份 episode record 的三个 landmark view**。

---

# 四、跨交易所场景中，“event time”必须保留多个坐标

在单 venue 中，可以用：

* trigger 后第 1 个事件；
* trigger 后第 5 个事件；
* trigger 后第 20 个事件。

但双 venue 中，一个单一 event index 会有问题。

假设 Binance 的消息更新频率远高于 Hyperliquid，那么：

```text
trigger 后第 20 个全局事件
```

可能实际上是：

```text
Binance 19 个事件
Hyperliquid 1 个事件
```

另一个 episode 则可能反过来。它们虽然都有 (k=20)，但并不处于相同的双市场响应阶段。

因此建议同时保存三套坐标：

## 1. 真实时间坐标

[
u_t=t-\tau_i^{cand}
]

单位是服务器 monotonic clock 上的微秒。

这是跨 venue actionability 的主坐标。

## 2. Binance venue-local event time

[
u_B
===

N_B(t)-N_B(\tau_i^{cand})
]

## 3. Hyperliquid venue-local event time

[
u_H
===

N_H(t)-N_H(\tau_i^{cand})
]

所以完整响应不是简单的：

[
R_i(u)
]

而更接近：

[
R_i(u_t,u_B,u_H)
]

实际存储不需要构造三维密集张量，可以保留原始事件流，再生成不同视图：

```text
clock-time view:
  50µs, 100µs, 250µs, 500µs, 1ms, 5ms ...

Binance-event view:
  B+1, B+3, B+5, B+10 ...

Hyperliquid-event view:
  H+1, H+3, H+5, H+10 ...
```

在跨交易所 maker 决策中：

> 真实微秒时间是执行主坐标；venue-local event count 是市场动力学的补充坐标。

---

# 五、(S^{pre}) 不应只是 trigger 前的一张盘口快照

更准确的定义是：

[
S_i^{pre}
=========

\phi
\left(
X_{[\tau_i^{cand}-L,\tau_i^{cand})}
\right)
]

也就是说，(S^{pre}) 是 trigger 前一段历史的表示，而不只是：

[
X(\tau_i^-)
]

因为同样的跨所 gap：

[
b_B-a_H=1\text{ tick}
]

可能来自完全不同的路径：

### Case A：Binance 刚刚上移

```text
Binance bid 上移
Hyperliquid 尚未响应
gap 刚形成
```

### Case B：Hyperliquid ask 刚刚下移

```text
Hyperliquid ask 下移
Binance 没有变化
gap 刚形成
```

### Case C：gap 已经维持很久

```text
两边都没动
gap 已存在 20ms
可能只是稳定 basis 或 feed anomaly
```

三个 case 的瞬时 spread 相同，但未来收敛机制完全不同。

因此跨 venue 的 (S^{pre}) 至少要包括：

* gap 当前水平；
* gap 持续时间；
* gap 最近变化率；
* 是哪一个 venue 产生了最后一次价格移动；
* 两边最新 feed age；
* 两边 spread 和 depth；
* source/target 的 OFI；
* source/target 最近 aggressive flow；
* 当前合约 basis 或慢速价差基线；
* own quote 和 target best 的相对位置；
* own order age 和 inventory。

Trigger 解决了“时间相位对齐”，而 (S^{pre}) 解决的是“历史路径与环境可比”。

原始材料中强调，相同的 trigger 后响应只有在前置状态足够相似时才能比较；否则看起来相同的 refill 可能有完全不同的市场意义。

---

# 六、跨所价差的 (R) 必须分解“通过谁收敛”

定义：

[
g_{B\rightarrow H}(u)
=====================

b_B(u)-a_H(u)
]

那么：

[
\Delta g(u)
===========

\Delta b_B(u)-\Delta a_H(u)
]

但只观察：

[
g(u)\rightarrow0
]

还不够，因为价差收敛可能通过两种完全相反的路径完成。

## Target catch-up

例如 Binance 出现向上冲击：

[
a_H(u)\uparrow
]

Hyperliquid 上移追赶 Binance。

这意味着 Hyperliquid ask maker 面临 adverse-selection 风险，`CANCEL ask` 可能有价值。

## Source reversion

[
b_B(u)\downarrow
]

Binance 冲击很快反转，Hyperliquid 不需要移动。

这时取消 Hyperliquid ask 可能只是错过一次良性成交。

所以 (R) 中不应只保存：

* gap 是否收敛；
* 收敛用了多久。

还必须保存：

[
R^{cross}
=========

\left(
\text{target catch-up path},
\text{source reversion path},
\text{joint movement path}
\right)
]

可以定义 competing events：

[
T_{\mathrm{target}}
===================

\text{target 首次沿 source shock 方向移动的时间}
]

[
T_{\mathrm{source-revert}}
==========================

\text{source 首次明显反转的时间}
]

[
T_{\mathrm{own-fill}}
=====================

\text{自己的危险侧挂单成交时间}
]

[
T_{\mathrm{cancel-effective}}
=============================

\tau^{effective}
]

真正影响 maker action 的概率是：

[
P\left(
T_{\mathrm{cancel-effective}}
<
T_{\mathrm{own-fill}}
\wedge
T_{\mathrm{target}}
\mid
\mathcal F_{\tau^{decision}}^{obs}
\right)
]

以及：

[
P\left(
T_{\mathrm{source-revert}}
<
T_{\mathrm{target}}
\mid
\mathcal F_{\tau^{decision}}^{obs}
\right)
]

这比单纯预测“gap 会不会收敛”更接近实际 EV。

---

# 七、你提到的 Palm distribution 是对的，但它是“每个事件”的分布

可以把 trigger 建模为一个带 mark 的点过程：

[
N_T(dt,dm)
]

其中 mark (m_i) 包括：

[
m_i=
(
venue,
side,
p_0,
Q_0,
I_i,
burst\ duration,
levels\ swept,
quality
)
]

那么你真正研究的是带 mark、带前置状态条件的 Palm distribution：

[
P^0
\left(
\widetilde X_i
\mid
S_i^{pre}=s,
M_i=m
\right)
]

它回答的是：

> 从一次典型 qualifying trigger 的视角看，前后市场路径如何分布？

但要注意，Palm distribution 给的是**per-event distribution**，不是单位时间策略收益。

假设某类状态中的 trigger 非常有利，但每个月只出现一次；另一类 edge 稍小，却每天出现数千次。仅比较 per-event EV 会误判策略价值。

单位时间收益还需要 trigger intensity：

[
\lambda_T(s)
]

整体收益率大致是：

[
\text{EV rate}
==============

\int
\lambda_T(s)
,
\mathbb E[
\Delta EV
\mid
T,s
]
,d\mu(s)
]

因此最终要同时报告：

* per-trigger EV；
* triggers per hour/day；
* deployed capital usage；
* cancel/action rate；
* per-unit-time PnL；
* tail event concentration。

Palm distribution 适合定义 episode family，但它本身不等于完整策略收益过程。

---

# 八、maker 的最终输入不是 ((S,T))，而是决策时信息状态

定义：

[
H_i^{d}
========

\left(
S_i^{pre},
T_i^{0:d_i},
R_i^{0:d_i},
S_i^{own},
Q_i^{data}
\right)
]

其中：

* (S^{pre})：trigger 前双 venue 状态；
* (T^{0:d})：截至 decision 已观察到的冲击路径；
* (R^{0:d})：截至 decision 已观察到的 target/source response；
* (S^{own})：自己的订单、queue estimate、inventory；
* (Q^{data})：feed age、gap、node health 等数据质量。

maker 的 policy 是：

[
\pi^*(H_i^d)
============

\arg\max_a
Q(a\mid H_i^d)
]

其中：

[
Q(a\mid H_i^d)
==============

\mathbb E
\left[
U(O_i^{own},O_i^{market})
\mid
H_i^d,
do(A_i=a)
\right]
]

这里使用 (do(A=a)) 是为了明确：

> 我们需要的是采取某个动作的反事实价值，而不只是历史中在某种状态下碰巧采取了什么动作。

在小规模 canary 下，如果自己的挂单对公共市场影响可以忽略，可以近似分解为：

[
P(
R^{future},
O^{market},
O^{own}
\mid
H^d,A
)
\approx
P(
R^{future},
O^{market}
\mid
H^d
)
,
P(
O^{own}
\mid
R^{future},H^d,A
)
]

这对应两个模型：

## 市场转移模型

[
P(
R^{future},O^{market}
\mid
H^d
)
]

回答：

* target 会不会追价；
* source 会不会反转；
* gap 如何收敛；
* target queue 如何变化。

## 执行响应模型

[
P(
O^{own}
\mid
R^{future},H^d,A
)
]

回答：

* KEEP 是否成交；
* CANCEL 是否及时生效；
* cancel race 是否成交；
* fill 后 markout；
* missed favorable fill；
* fees、inventory 和 re-entry cost。

公共聚合 L2 只能支持市场聚合响应，无法证明不同 episode 中的行为来自同一个 maker；有了自己的订单回报，才能可靠地建立自己的 (S\rightarrow A\rightarrow O) 链路。

---

# 九、episode 应该采用“逐步追加”，而不是事后一次性生成

这套定义特别适合 event-sourcing。

一个 episode 的生命周期可以是：

```text
EpisodeOpened
  at τ_candidate

ThresholdCrossed
  at τ_theta

TriggerConfirmed / TriggerRejected
  at τ_confirm

DecisionMade
  at τ_decision

ActionEffective / ActionFailed / CancelRaceFill
  at τ_effective or fill time

EpisodeCensored / EpisodeClosed
  at horizon H
```

每条 episode field 除了值，还应保存：

```text
value
observed_at
source_event_id
source_book_version
calculation_version
```

例如：

```text
impact_ratio = 0.71
impact_ratio_observed_at = 123456789ns
```

这允许系统自动检查：

[
observed_at(feature)
\le
\tau^{decision}
]

只要某个策略 feature 的 `observed_at` 晚于 decision，它就不能进入该次决策。

这会比人工检查特征有没有未来数据可靠得多。

---

# 十、必须加入 censoring 和 episode contamination

并不是每个 episode 都能观察到完整的 (R,O)。

例如：

* feed 中断；
* WebSocket reconnect；
* snapshot reset；
* Hyperliquid node 落后；
* Binance sequence gap；
* 订单因 inventory override 被其他策略撤销；
* 下一个 shock 在当前 episode 尚未结束时出现；
* source 和 target 同时发生独立冲击；
* 到达研究 horizon 前策略进程重启。

因此 episode 还应包含：

[
C_i=
(
censor\ time,
censor\ reason,
overlap\ flags,
data\ quality
)
]

不能简单把这些 episode 删除，因为数据中断和高波动往往相关。删除它们可能系统性地保留平静市场、排除最危险市场。

对于未完整观察的 response time，可以使用 survival/competing-risk 方式处理，而不是强制标记成“未发生”。

---

# 十一、同一个 burst 不要因为多个阈值而生成多个独立 episode

如果一个 burst 依次越过：

[
0.25,\quad0.50,\quad0.75,\quad1.00
]

不应该生成四个高度重叠、被当作独立样本的 episode。

正确做法是：

* 一个 burst 对应一个 `episode_id`；
* 保存一条持续演化的 severity path：

[
I_i(u)
======

\frac{V_{p_0,i}(u)}{Q_{0,i}}
]

* 保存每个阈值的 crossing time：

[
\tau_i^{0.25},
\tau_i^{0.50},
\tau_i^{0.75},
\tau_i^{1.00}
]

这样可以在同一 episode 内研究：

* 冲击剂量如何演变；
* 策略在不同阈值时行动会有怎样的提前量；
* 等待更高置信度是否已经失去 actionability。

还要定义 shock cooldown 或 episode-merging contract，避免同一持续冲击被切成十几个相邻 cases。

---

# 十二、最终可以把理论核心再压缩成四层

你现在的表述可以进一步整理为：

## 第一层：随机时钟

[
\tau_i
======

\text{stopping time defined on server-observed information}
]

Trigger 将 calendar time 转换为 event-relative coordinates。

## 第二层：带条件的 episode family

[
S_i^{pre}
=========

\phi(X_{[\tau_i-L,\tau_i)})
]

[
M_i=
\text{trigger mark and severity path}
]

Trigger 负责时间配准和样本选择；(S^{pre}) 与 trigger mark 负责可比性。

## 第三层：在线条件转移分布

[
P(
R^{future},O^{market}
\mid
S^{pre},T^{prefix},R^{prefix}
)
]

历史 case library 用于估计当前可观测信息下的未来市场分布。

## 第四层：动作反事实价值

[
Q(a\mid H^d)
============

\mathbb E[
U(O^{own})
\mid
H^d,do(A=a)
]
]

maker 在 `KEEP/CANCEL/...` 中选择风险调整后价值最大的动作。

因此最完整的一句话是：

[
\boxed{
\text{Trigger 定义随机时钟；
状态和 trigger mark 定义可比较的条件邻域；
response prefix 更新未来市场分布；
真实动作反馈识别不同 maker action 的反事实价值。}
}
]

这比单纯的：

[
(S,T)\rightarrow P(R,O)
]

更精确，因为它明确处理了：

* live 信息集；
* candidate 与 confirmation 的时差；
* trigger 后逐渐到达的信息；
* 跨交易所不同事件时钟；
* 自己动作造成的执行差异；
* false candidates；
* cancel race；
* 数据 censoring；
* per-event alpha 与 per-time PnL 的区别。

从工程角度看，下一节点已经非常明确：先实现一个**只追加、不回写未来信息的 Episode State Machine**，并让 live、replay 和研究过程共同消费同一份 episode lifecycle。这样理论定义会真正变成系统约束，而不是只停留在研究文档里。
