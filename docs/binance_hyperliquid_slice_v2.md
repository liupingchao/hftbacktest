你的理解**基本成立**。更准确地说，market maker 面对的不是：

[
(S,T)\longrightarrow \text{一个确定结果}
]

而是：

[
(S,T)\longrightarrow
P(R,O\mid S,T)
]

也就是：在当前可观测状态和某个明确冲击下，市场随后可能以多种方式响应，每种响应又对应不同的执行和价格结果。maker 做的是**条件概率分布上的决策优化**，而不是寻找一条永远成立的确定规则。

不过，这个框架要真正变成可执行策略，还需要三个关键修正：

1. 完整的 (R) 在决策时通常还没有发生，只能使用 (R) 的早期 prefix；
2. 执行结果 (O) 依赖于你采取的动作，所以 episode 中必须加入 (A)；
3. “尽可能多地收集 cases”不等于简单累积历史，要保证 cases 可比、近期、数据质量可靠，并且具有足够的 action coverage。

---

# 一、为什么相同 (S,T) 下，(R,O) 天然是概率分布

市场中存在大量不可观测变量：

* 其他 maker 的库存；
* 其他 venue 的订单；
* 隐藏和 iceberg 流动性；
* 尚未到达你服务器的信息；
* 不同参与者的交易意图；
* 清算、套利和对冲流；
* 各交易所不同的网络与撮合延迟。

可以把这些隐藏信息表示为 (Z)。真实过程更接近：

[
P(R,O\mid S,T)
==============

\int
P(R,O\mid Z,S,T)
P(Z\mid S,T),dZ
]

你看到的 (S) 只是隐藏状态 (Z) 的不完整投影。

因此，即便两个 episode 的 (S) 和 (T) 看起来非常接近，后面也可能分别出现：

* source venue 快速 refill；
* source venue 队列完全耗尽；
* target venue 先移动；
* target venue 不动；
* 跨所价差快速收敛；
* 跨所价差进一步扩大；
* 你的 maker order 被成交；
* 你的 maker order没有成交。

这并不表示规律不存在，而是说明规律的形式是：

> **条件响应分布，而不是确定映射。**

原始文章强调的“只在相似市场状态下比较市场响应”正是这个意思。

---

# 二、跨交易所价差是强结构事实，但要区分四种 alpha

设：

* (b_B,a_B)：Binance best bid/ask；
* (b_H,a_H)：Hyperliquid best bid/ask。

两个方向的 raw cross spread 是：

[
g_{B\rightarrow H}=b_B-a_H
]

[
g_{H\rightarrow B}=b_H-a_B
]

例如：

[
b_B>a_H
]

表示在你当前服务器看到的市场 view 中，Binance bid 高于 Hyperliquid ask。

这个结构肯定有信息，但“肯定有 alpha”最好进一步拆成四层。

## 1. 信息 alpha

跨所价差对其中一个 venue 随后的价格移动有预测力：

[
P(\Delta mid_H>0\mid b_B-a_H)
]

这通常是最容易验证的一层。

## 2. maker-risk alpha

跨所价差可以识别某一侧 maker quote 是否 stale。

例如：

[
b_B-a_H>0
]

意味着 Hyperliquid ask 很可能处于危险位置。即使你来不及跨所套利，也可能来得及取消 Hyperliquid ask，避免 adverse fill。

这和你的 maker 策略最直接相关。

## 3. execution alpha

你不仅预测对了，还能在 target venue 发生移动前完成：

* 决策；
* cancel/send；
* order acknowledgement；
* 实际撤单或成交。

它取决于：

[
t_{\mathrm{target\ response}}
-----------------------------

t_{\mathrm{action\ effective}}
]

而不仅是价格相关性。

## 4. arbitrage alpha

只有两边在你的订单到达时仍可成交，并扣除：

* taker fees；
* maker fees/rebates；
* slippage；
* partial fill；
* hedge risk；
* inventory funding；
* cancel race；

之后仍为正，才是可锁定的跨所套利。

因此更严谨的说法是：

> 跨所价差是高价值的结构性信息变量；但 raw observed spread 不等于 guaranteed executable profit。

服务器看到的正价差可能有一部分来自两条 feed 的异步到达。对 maker 来说，这也不一定没用：即使不是可双腿成交的套利，它仍可能告诉你 target quote 已经过时。

---

# 三、在第一条 slice 里，跨所价差应该放在哪里

你目前的第一条 vertical slice 仍然是：

[
T:
\quad
\frac{
\text{aggressive trade volume}
}{
\text{pre-trigger best queue}
}
\ge\theta
]

我建议**不要同时把跨所价差阈值也变成第二个 trigger**。否则你很快会混淆：

* 是 aggressive shock 有效；
* 是 cross spread 有效；
* 还是二者组合才有效。

在这条 slice 中，跨所价差分别进入 (S) 和 (R)。

## Trigger 前的跨所价差属于 (S^{pre})

例如：

[
g_{B\rightarrow H}^{pre}
========================

b_B^{pre}-a_H^{pre}
]

以及：

* cross-mid difference；
* 两边 spread；
* 两边 feed age；
* 两边可见 depth；
* 两边短期 OFI；
* 两边最近价格移动；
* basis 和 tick-normalized gap。

它表示 shock 发生之前，两个 venue 已经处于什么相对位置。

## Trigger 后的跨所价差路径属于 (R)

例如：

[
g_{B\rightarrow H}(\tau),
\qquad
\tau\in[0,H]
]

你要观察：

* gap 是否扩大；
* 哪个 venue 先移动；
* target venue 用多少微秒或事件数追上；
* gap 是通过 Binance 回撤还是 Hyperliquid 上移而收敛；
* target ask 是先被成交，还是先撤掉；
* target depth 是否在价格移动前撤退。

因此，当前 slice 的问题可以写成：

[
P\left(
R_{\mathrm{Binance}},
R_{\mathrm{Hyperliquid}},
g(\tau),
O
\mid
S^{pre}*{B,H},
T*{\mathrm{source}}
\right)
]

以后可以单独做第二条 slice：

[
T_{\mathrm{cross-spread}}
:
\quad
g^{net}>\theta_g
]

那时跨所价差本身才成为 trigger。

---

# 四、原来的 episode 定义需要加入“决策时刻”和“自己的动作”

你现在的定义是：

[
E_i=
(
S_i^{pre},
T_i,
R_i,
O_i
)
]

它适合做**市场行为研究**。

但它还不是完整的 maker decision episode，因为完整 (R_i) 是 trigger 之后才逐渐发生的，而你的 maker 必须在 (R_i) 尚未结束时决策。

更适合生产策略的定义是：

[
\boxed{
E_i=
\left(
S_i^{pre},
T_i,
R_i^{0:\tau_d},
A_i,
R_i^{\tau_d:H},
O_i
\right)
}
]

其中：

* (S_i^{pre})：trigger 前的双 venue 状态；
* (T_i)：source venue 的 aggressive queue shock；
* (R_i^{0:\tau_d})：到决策时刻为止已经看到的早期响应；
* (A_i)：你采取的动作；
* (R_i^{\tau_d:H})：决策之后市场继续怎样响应；
* (O_i)：市场结果和自己的执行结果。

这里 (\tau_d) 是明确的 decision timestamp。

例如：

```text
t0        Binance aggressive queue shock candidate
t0+200µs  Hyperliquid 尚未移动
t0+250µs  strategy 做出 KEEP/CANCEL 决策
t0+400µs  cancel 请求到达交易所
t0+700µs  Hyperliquid ask 被打或向上移动
```

在 (t_0+250\mu s) 做决策时，你只能使用：

[
S^{pre},T,R^{0:250\mu s}
]

不能使用完整的：

[
R^{0:H}
]

否则会产生未来信息泄漏。

---

# 五、最好再把 (O) 分成市场结果与执行结果

建议定义：

[
O_i=
(
O_i^{market},
O_i^{exec}
)
]

## 市场结果 (O^{market})

包括：

* target mid markout；
* source mid markout；
* cross-spread convergence time；
* 哪个 venue 先移动；
* target BBO 是否移动；
* target queue 是否耗尽；
* 10/50/100/500ms 后相对价格；
* 后续是否发生反向 shock。

这些结果即使你没有下单，也可以观测。

## 执行结果 (O^{exec})

包括：

* own order 是否成交；
* fill timestamp；
* cancel-race fill；
* cancel acknowledgement；
* realized spread；
* post-fill markout；
* fees/rebates；
* hedge cost；
* inventory change；
* episode-level PnL。

公共 L2 只能告诉你聚合盘口怎样响应，不能知道不同 episode 是否由同一个 maker 驱动。

但你有自己的订单回报后，就可以真正分析：

[
S
\rightarrow
A
\rightarrow
O^{exec}
]

而不再需要从公共 L2 猜测自己的执行结果。

---

# 六、完整的双交易所 episode schema

对于一次 Binance aggressive buy shock、目标是在 Hyperliquid 做 maker，可以定义：

## (S^{pre})：trigger 前状态

[
S^{pre}=
\left(
S_B,
S_H,
S_{cross},
S_{own}
\right)
]

### Binance 状态 (S_B)

* best bid/ask；
* top-N depth；
* ask best queue (Q_0)；
* L1/L5 imbalance；
* OFI；
* recent aggressive flow；
* spread；
* short volatility；
* feed age。

### Hyperliquid 状态 (S_H)

* best bid/ask；
* top-N depth；
* target ask queue；
* target OFI；
* target recent trades；
* spread；
* target last price-move age；
* feed age。

### 跨 venue 状态 (S_{cross})

* (b_B-a_H)；
* (b_H-a_B)；
* mid difference；
* tick-normalized basis；
* 哪个 venue 最近先动；
* 两边 feed 到达时间差；
* 当前 actionability margin 估计。

### 自己的状态 (S_{own})

* 是否有 Hyperliquid ask resting；
* quote price 和 best 的距离；
* order age；
* estimated queue position；
* size；
* inventory；
* pending cancel/order；
* 最近的 order latency。

---

## (T)：冲击结构

[
T=
(
v_{\mathrm{source}},
side,
p_0,
Q_0,
V_{p_0},
I,
duration,
levels\ swept
)
]

例如：

```text
source venue        = Binance
direction           = aggressive buy
pre-trigger ask     = 100000
pre-trigger queue   = 12 BTC
volume at p0        = 8.4 BTC
impact ratio        = 0.70
burst duration      = 400µs / feed microbatch
levels swept        = 1
```

---

## (R^{0:\tau_d})：早期可见响应

例如：

* Binance ask 是否消失；
* Binance 是否出现同价 refill；
* Binance BBO 是否上移；
* (b_B-a_H) 是否转正或继续扩大；
* Hyperliquid ask 是否撤单；
* Hyperliquid trades 是否开始转为 aggressive buy；
* Hyperliquid mid 是否尚未移动；
* 两边经过了多少事件和多少微秒。

这是真正进入在线 decision model 的 response prefix。

---

## (A)：自己的动作

第一版保持极简：

[
A\in
{
KEEP,
CANCEL_RISK_SIDE
}
]

以后再扩展：

[
A\in
{
KEEP,
CANCEL,
RESIZE,
WIDEN,
REPRICE,
HEDGE
}
]

---

## (R^{\tau_d:H}) 与 (O)

决策后继续观察：

* target ask 是否先被打；
* cancel 是否先完成；
* target mid 是否上移；
* cross spread 如何收敛；
* source 是否回撤；
* fill 后 markout；
* 不成交时错过了多少有利 spread capture。

---

# 七、case library 的作用不是找“完全相同”的历史

高维市场状态中，几乎不存在两个完全相同的 (S,T)。

实际做法是，在当前 query：

[
x_t=
(
S_t^{pre},
T_t,
R_t^{0:\tau_d}
)
]

附近寻找一个局部 case neighborhood：

[
\mathcal N(x_t)
===============

\left{
i:
d(x_t,x_i)<\epsilon
\right}
]

然后给历史 cases 加权：

[
w_i
\propto
K\left(d(x_t,x_i)\right)
\cdot
w_{\mathrm{recency},i}
\cdot
w_{\mathrm{quality},i}
\cdot
w_{\mathrm{regime},i}
]

这里的权重至少反映：

* 状态相似程度；
* 数据新旧；
* feed 是否健康；
* 是否属于同样的 source→target 方向；
* latency regime 是否相似；
* volatility 是否相似；
* 自己的 order state 是否相似。

最后估计：

[
\widehat P
\left(
R^{future},O
\mid
x_t,A=a
\right)
]

重点不是只估计均值，而是保留整个分布：

* target 上移概率；
* target 上移时间分布；
* own fill probability；
* cancel-race probability；
* favorable fill probability；
* adverse markout 分布；
* tail loss；
* missed-fill opportunity cost。

---

# 八、“尽可能多的 cases”要改成“尽可能多的有效覆盖”

历史 case 数量当然重要，但它不是越多越好。

三年前、不同服务器地区、不同 fee tier、不同 feed schema 的 case，可能比没有 case 更危险。

真正需要最大化的是：

[
\text{effective support}
========================

\text{数量}
\times
\text{相似度}
\times
\text{近期性}
\times
\text{数据质量}
\times
\text{action coverage}
]

可以计算局部有效样本量：

[
N_{\mathrm{eff}}
================

\frac{
\left(\sum_i w_i\right)^2
}{
\sum_i w_i^2
}
]

当 (N_{\mathrm{eff}}) 太低时，策略不应该假装自己知道答案，而应该回退到 baseline：

```text
低置信度 → baseline maker policy
中等置信度 → reduce size
高置信度 → cancel/keep overlay
```

这也是 case-based 方法应对非平稳性的关键：不是一股脑把所有旧数据扔进一个巨型训练集，而是判断**当前 episode 究竟有哪些可信的历史邻居**。

---

# 九、从概率分布到 maker action

在决策时刻，策略真正要比较的是每个 action 的条件价值：

[
Q(a\mid x_t)
============

\mathbb E[
U(O)
\mid
x_t,A=a
]
]

选择：

[
a^*
===

\arg\max_a
Q(a\mid x_t)
]

对于 `KEEP` 和 `CANCEL`，可以写成：

## KEEP

[
EV_{\mathrm{keep}}
==================

P(F\mid x_t,\mathrm{keep})
\left[
\text{spread capture}
+
\text{rebate}
-------------

\mathbb E(
\text{adverse markout}
\mid F,x_t
)
\right]
-------

C_{\mathrm{inventory}}
]

## CANCEL

[
EV_{\mathrm{cancel}}
====================

*

## C_{\mathrm{missed\ favorable\ fill}}

P(F_{\mathrm{cancel-race}})
\mathbb E[
\text{adverse markout}\mid F_{\mathrm{cancel-race}}
]
-

C_{\mathrm{reentry}}
]

然后：

[
CANCEL
\quad\text{iff}\quad
EV_{\mathrm{cancel}}

>

EV_{\mathrm{keep}}
]

这比简单规则：

```text
cross spread > 0 → cancel
```

更完整。

因为某些时候即使 cross spread 为正：

* 你的 ask 距离 best 很远；
* 当前 inventory 很空，需要卖出；
* target venue 已经开始 refill；
* source shock 很快反转；
* cancel latency 太长；
* 你的订单本来就几乎不会成交；

此时 cancel 未必有价值。

---

# 十、一个非常关键的识别问题：历史 cases 必须包含不同动作

假设过去所有 qualifying episode 中，你都选择了 `CANCEL`。

那么你可以观察：

[
P(O\mid x,CANCEL)
]

但无法直接知道：

[
P(O\mid x,KEEP)
]

因为没有 counterfactual。

同样，如果过去永远 `KEEP`，也不知道 cancel 是否更好。

因此，case library 不仅需要大量市场 episode，还需要一定的 action exploration：

[
P(A=a\mid x)>0
]

第一版可以采用很小风险的 deterministic canary：

```text
qualifying episodes:
  90% baseline
  10% treatment
```

或者在安全状态内：

```text
hash(episode_id) → KEEP / CANCEL
```

并记录 assignment probability。之后才能比较同类状态下：

[
\mathbb E[O\mid x,A=KEEP]
]

和：

[
\mathbb E[O\mid x,A=CANCEL]
]

否则你得到的只是市场预测模型，还不是 policy learning。

---

# 十一、fill-conditioned 数据会产生严重偏差

不能只收集“被成交的 maker orders”。

因为成交本身不是随机的：

[
P(S,T,R\mid fill)
\neq
P(S,T,R)
]

最危险、最不利的市场状态往往更容易让 maker order 成交，这就是 adverse-selection selection bias。

所以每个 trigger 都必须进入 episode store，包括：

* 有 resting order、最后成交；
* 有 resting order、成功撤单；
* 有 resting order、未成交也未撤；
* 没有 resting order；
* treatment cancel；
* control keep；
* cancel-race fill；
* 本可有利成交但被 cancel 错过。

否则模型会错误地认为“成交样本的 markout 就是所有挂单机会的 markout”。

---

# 十二、用一个具体例子串起来

当前服务器观察到：

[
I_B
===

\frac{
V_{\mathrm{aggressive\ buy\ at\ Binance\ ask}}
}{
Q_{\mathrm{Binance\ ask},0}
}
=0.72
]

同时：

[
b_B-a_H=1.5\text{ ticks}
]

Hyperliquid ask 仍未移动，你在该 ask 有一个 resting maker order。

决策输入是：

[
x_t=
(
S^{pre}_B,
S^{pre}_H,
g^{pre},
T_B,
R^{0:\tau_d},
S^{own}
)
]

从历史 case neighborhood 得到：

[
P(
\text{HL ask 在 5ms 内被打}
\mid x_t
)
=0.64
]

[
P(
\text{HL mid 在 10ms 内上移}
\mid x_t
)
=0.71
]

[
E[
\text{50ms markout}
\mid
fill,x_t
]
=-1.8\text{ ticks}
]

[
P(
\text{cancel 在 fill 前生效}
\mid x_t
)
=0.82
]

但同时可能有：

[
P(
\text{cancel 后错过 favorable fill}
\mid x_t
)
=0.09
]

策略不是因为“Binance 涨了”就机械 cancel，而是比较：

[
EV_{\mathrm{keep}}
\quad\text{和}\quad
EV_{\mathrm{cancel}}
]

然后根据当前 inventory 和风险偏好做出动作。

这就是 episode library 真正转化为 market-maker policy 的方式。

---

# 十三、这个框架最凝练的表达

你现在的理解可以升级成：

[
\boxed{
\begin{aligned}
&\text{用机制明确的 trigger 对齐历史 cases}\
&\rightarrow
\text{用双 venue 前置状态找到局部可比案例}\
&\rightarrow
\text{根据当前已观察到的 response prefix 更新未来分布}\
&\rightarrow
\text{分别估计不同 maker action 下的执行结果分布}\
&\rightarrow
\text{选择风险调整后 EV 最大的 action}\
&\rightarrow
\text{将真实订单反馈重新写回 case library}
\end{aligned}
}
]

更形式化一点：

[
\boxed{
\left(
S^{pre},
T,
R^{prefix},
A
\right)
\longrightarrow
\mathcal D
\left(
R^{future},
O^{market},
O^{exec}
\right)
}
]

你要学习的不是一个传统意义上的单一预测函数，而是一个**局部、非平稳、可持续更新的随机转移核**。

maker policy 则是：

[
\boxed{
\pi^*(A\mid S,T,R^{prefix})
===========================

\arg\max_A
\mathbb E[
U(O^{exec})
\mid
S,T,R^{prefix},A
]
}
]

所以你的直觉可以总结为一句话：

> **trigger 负责把可比较的市场事件对齐，case library 负责提供条件未来分布，而 maker policy 负责在该分布上选择最优动作。**

唯一必须补上的，是：不能只学习“市场接下来会怎样”，还必须学习“在市场可能这样发展的前提下，我采取不同动作分别会怎样”。


是的，这次的细化抓到了 episode 方法里最重要的一层：

> **Trigger 的首要作用不是提供一个普通特征，而是把连续时间过程转换成统一的 event-time 坐标系。**

没有 trigger，不同市场片段只是发生在不同时间、不同阶段的连续序列，无法直接平均或比较。

**数学表达**

设双交易所连续市场过程为：

\[
X(t)=\left(X_B(t),X_H(t)\right)
\]

Trigger detector 在一组时刻产生事件：

\[
\{\tau_1,\tau_2,\ldots,\tau_n\}
\]

对每个事件重新定义相对时间：

\[
u=t-\tau_i
\]

那么第 \(i\) 个对齐后的 episode 是：

\[
\widetilde X_i(u)=X(\tau_i+u),
\qquad u\in[-L,H]
\]

此时所有 episode 都满足：

```text
u < 0   trigger 前
u = 0   相同语义的 trigger 发生
u > 0   trigger 后的市场响应
```

这样才能比较：

- trigger 后多少微秒 target 开始撤单；
- 多少微秒后跨所价差收敛；
- target 是先成交、先撤单还是先移动；
- 哪类前置状态对应更快的 adverse response；
- cancel 是否能在 adverse event 前生效。

这实际上是在估计一个 **event-conditioned distribution**，严格一点说，接近于对 trigger point process 的 Palm distribution：

\[
P\left(\widetilde X(u),u\in[-L,H]\mid \text{trigger at }u=0\right)
\]

**Trigger 有三个不同作用**

1. **时间配准**

把不同日期、不同 session 的序列拉到统一原点。

2. **样本选择**

只有满足同一 trigger contract 的市场片段才能进入这个 episode family。

3. **条件信息**

trigger 的方向、严重程度、队列消耗比例、成交结构等仍然是预测未来分布的输入变量。

所以 trigger 既是坐标原点，也是条件变量，但这两个作用不能混为一谈。

**统一原点还不等于完全可比**

Trigger 解决的是时间相位问题，`Spre` 和 `T` 解决的是状态可比问题。完整过程是：

```text
trigger alignment
-> state/trigger similarity
-> conditional response distribution
-> action-dependent outcome distribution
```

即：

\[
\text{时间对齐：}\quad u=t-\tau_i
\]

\[
\text{状态对齐：}\quad
X_i=(S_i^{pre},T_i)
\]

\[
\text{结果估计：}\quad
P(R_i,O_i\mid X_i)
\]

两个 episode 即使都在 `u=0` 发生 shock，如果一个发生在高波动、宽 spread、旧行情环境，另一个发生在低波动、新鲜行情环境，它们仍不能直接当作同类样本。

**必须区分五个时间点**

当前项目里不应该只保存一个 `trigger_ts`，而应明确：

```text
τ_start       burst 开始
τ_candidate   触发阈值首次达到
τ_confirm     trigger 被确认
τ_decision    策略作出 KEEP/CANCEL
τ_effective   cancel 实际生效
```

建议主 episode 使用：

\[
u=0\equiv\tau_{candidate}
\]

因为它是最早的因果 trigger 原点。然后保存：

\[
d_{confirm}=\tau_{confirm}-\tau_{candidate}
\]

\[
d_{decision}=\tau_{decision}-\tau_{candidate}
\]

\[
d_{effective}=\tau_{effective}-\tau_{candidate}
\]

这样既能对齐完整响应路径，也不会把确认时间、决策时间和 trigger 原点混在一起。

v1 在 `Confirmed` 后行动没有问题，但必须写成：

\[
A_i=
\pi\left(
S_i^{pre},
T_i^{0:d_{decision}},
R_i^{0:d_{decision}}
\right)
\]

不能因为 episode 以 Candidate 为原点，就假装策略在 Candidate 时已经知道后续 confirmation。

**更完整的 episode 定义**

我建议把你的定义进一步整理为：

\[
E_i=
\left(
\mathcal T_i,
S_i^{pre},
T_i^{0:d_i},
R_i^{0:d_i},
A_i,
R_i^{d_i:H},
O_i^{market},
O_i^{own}
\right)
\]

其中：

- \(\mathcal T_i\)：所有时间锚点；
- \(S_i^{pre}\)：trigger 前双 venue 状态；
- \(T_i^{0:d_i}\)：决策前已经观察到的 trigger 路径；
- \(R_i^{0:d_i}\)：决策前 target 的早期响应；
- \(A_i\)：KEEP 或 CANCEL；
- \(R_i^{d_i:H}\)：决策后市场响应；
- \(O_i^{market}\)：未来市场价格结果；
- \(O_i^{own}\)：自己的成交、费用、库存与 PnL。

**对当前路线的直接意义**

第一条 slice 继续以 aggressive queue shock 作为唯一 trigger 是正确的。跨交易所价差在这里承担两种角色：

- trigger 前的价差水平、残差和变化进入 `Spre`；
- trigger 后的价差收敛路径进入 `R`。

随后我们真正估计的是：

\[
P\left(
R,O^{market},O^{own}(a)
\mid
S^{pre},T
\right)
\]

最后比较：

\[
\Delta EV(S,T)
=
EV(CANCEL\mid S,T)
-
EV(KEEP\mid S,T)
\]

因此可以把你的核心认识总结为：

> **Trigger 把 calendar time 转换成 event time；`Spre` 和 `T` 再把已对齐的 episodes 划分成可比较的条件集合；market maker 最终从这些集合中估计不同动作的未来收益分布。**

这个表述已经可以成为整个 episode research architecture 的理论核心。
