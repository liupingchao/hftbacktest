这个场景里，motif 方法可以用，但需要先改一下目标定义：

> 不应直接从 L2 图形中寻找“某个做市商的策略”，而应先寻找**反复出现的流动性提供响应模式**，再判断这些模式是否像一套稳定的 maker policy。

因为普通 L2/MBP 数据把多个参与者的挂单聚合在同一价位。如果没有 order ID、participant ID、MPID 或自有订单回报，通常无法证明两次行为来自同一个 maker。你能高置信度发现的是：

* 挂单补充模式；
* 撤单和重新报价模式；
* 买卖盘偏斜模式；
* 成交后的补单模式；
* 面对不利订单流时的撤退模式；
* 这些模式对应的成交概率和后续 adverse selection。

因此，更准确的目标是：

> **从 L2+trades 中发现稳定、重复、具有做市特征的“隐含策略族”。**

---

# 一、HFT 场景中的 motif 是什么

图像场景中，motif 是重复出现的时空图案。

订单簿场景中，motif 是一段重复出现的：

[
\text{市场状态}
\rightarrow
\text{流动性响应}
\rightarrow
\text{成交及价格结果}
]

可以把一次候选 episode 表示为：

[
X_i=
\left[
S_i^{\mathrm{pre}},
R_i,
O_i
\right]
]

其中：

* (S^{\mathrm{pre}})：事件发生前的盘口状态；
* (R)：随后出现的挂单、撤单、成交和队列变化；
* (O)：最终成交、价差、价格移动和 adverse-selection 结果。

例如一个 motif 可能是：

```text
买方主动成交吃掉 70% 最优卖盘
→ 最优卖盘在很短时间内被快速补回
→ 买盘没有同步增加
→ spread 保持 1 tick
→ 中间价没有继续向上移动
```

这可以解释为一种：

> 卖方流动性提供者在主动买单冲击后快速补充 ask 的响应模式。

但它还不能自动证明这是同一家做市商。

---

# 二、不要直接对原始 L2 矩阵找 motif

假设每个时刻有前 10 档：

[
Q_t=
[
q^b_{t,1},\ldots,q^b_{t,10},
q^a_{t,1},\ldots,q^a_{t,10}
]
]

直接比较两个 L2 窗口会被很多无关因素主导：

* 标的正常成交量变化；
* 日内流动性季节性；
* 不同价格水平；
* spread 是 1 tick 还是 2 tick；
* 市场活跃速度不同；
* 一个窗口发生在买方冲击后，另一个发生在卖方冲击后；
* 队列总规模不同，但相对响应其实相同。

因此 motif 不应定义在“绝对盘口”上，而应定义在：

> **经过状态、价格、方向和流动性标准化之后的订单簿响应上。**

---

# 三、最关键的对齐

在订单簿 motif 中，“对齐”比图像中更重要。

## 1. 时间对齐：围绕触发事件

每个 episode 都选一个明确的事件作为 (\tau=0)。

例如：

* 一笔主动买单；
* 一笔主动卖单；
* 最优档被消耗超过某个比例；
* best bid 或 best ask 被完全吃掉；
* spread 从 1 tick 扩大到 2 tick；
* OFI 出现异常冲击；
* 某一侧出现大规模撤单；
* 中间价发生一个 tick 的跳变。

然后截取：

[
X_i=
[
t_i-T_{\mathrm{pre}},
t_i+T_{\mathrm{post}}
]
]

例如用事件时间：

```text
触发前 50 个盘口事件
触发后 100 个盘口事件
```

而不是固定 100 毫秒。

HFT 数据中，事件到达速度变化很大。事件时间通常比固定时钟时间更适合比较订单簿动力学；队列反应模型也是将订单到达、撤单和成交视为依赖当前盘口状态的事件过程。([arXiv][1])

但延迟本身如果是策略特征，就要另外保存：

```text
事件序号上的响应形状
+
真实微秒延迟
```

不能只保留事件时间而完全丢掉时钟时间。

---

## 2. 价格对齐：以当前 mid/best 为原点

不要使用绝对价格：

[
100.01,\ 100.02,\ 100.03
]

而使用相对 tick：

[
\ell=
\frac{p-p_{\mathrm{mid},0}}{\mathrm{tick}}
]

或者以触发时的 best bid/ask 为基准：

```text
Level 0：当前最优价
Level 1：距离最优价 1 tick
Level 2：距离最优价 2 ticks
```

这样不同日期、不同绝对价格的 episode 才能比较。

---

## 3. 买卖方向对齐

假设一个 episode 由主动买单触发，另一个由主动卖单触发。

如果策略是对称的，它们其实是镜像：

```text
主动买单 → ask 被消耗 → ask 补单
主动卖单 → bid 被消耗 → bid 补单
```

可以把所有 episode 转成统一方向：

```text
aggressive side
passive side
same side
opposite side
```

例如把主动卖单 episode 做镜像，使所有输入都变成：

> “主动流冲击被动卖方之后，盘口如何响应。”

这样同一类对称做市逻辑不会被拆成买方类和卖方类。

---

## 4. 流动性尺度对齐

绝对增加 100 手，在不同盘口里意义完全不同。

可以用：

[
\Delta \widetilde q_{t,\ell}
============================

\frac{\Delta q_{t,\ell}}
{\operatorname{median\ depth}_{\mathrm{local}}+\epsilon}
]

或者相对于触发前队列：

[
\Delta \widetilde q_{t,\ell}
============================

\frac{
q_{t,\ell}-q_{0,\ell}
}{
q_{0,\ell}+\epsilon
}
]

成交量也类似：

[
\widetilde v_t=
\frac{v_t}{\text{近期成交量尺度}}
]

这样 motif 描述的是：

> 补回了原队列的 80%，

而不是：

> 补了 500 手。

---

# 四、每个 episode 应该包含哪些通道

建议构造一个三维张量：

[
X_i[\tau,\ell,c]
]

其中：

* (\tau)：相对于触发事件的时间或事件序号；
* (\ell)：相对价位；
* (c)：特征通道。

## 盘口状态通道

* bid/ask 各档深度；
* spread；
* mid-price；
* microprice；
* queue imbalance；
* 多档 depth imbalance；
* OFI；
* 局部波动率；
* 近期成交方向和强度。

短时间价格变化与订单流不平衡之间的关系通常比单纯成交量更稳健，因此 OFI 应作为核心状态变量，而不能只看 trade volume。([arXiv][2])

## 事件流通道

最好区分：

* 新增 bid limit；
* 撤销 bid；
* 新增 ask limit；
* 撤销 ask；
* 主动买成交；
* 主动卖成交；
* best queue depletion；
* price level creation/removal。

如果只有聚合 L2，而没有逐订单消息，只能推断：

[
\Delta q
========

## \text{新增}

## \text{撤销}

\text{成交消耗}
]

结合 trades 可以分离一部分成交消耗，但通常不能完整识别每一笔新增和撤单。

## maker-like 响应通道

在原始通道上再构造更解释性的特征。

### 补单比例

[
RR=
\frac{
\text{冲击后新增被动量}
}{
\text{被消耗的被动量}+\epsilon
}
]

### 补单延迟

[
L_{\mathrm{refill}}
===================

t_{\mathrm{first\ refill}}-t_{\mathrm{trigger}}
]

同时记录微秒延迟和事件数延迟。

### 撤单不对称

[
CA=
\frac{
C_{\mathrm{same}}-C_{\mathrm{opposite}}
}{
C_{\mathrm{same}}+C_{\mathrm{opposite}}+\epsilon
}
]

### 报价偏斜

[
QS=
D_{\mathrm{bid}}-D_{\mathrm{ask}}
]

最好使用归一化深度。

### 报价持续性

* 补单后维持了多久；
* 是否很快再次撤销；
* 是否直到被成交；
* 是否跟随 mid-price 重挂。

---

# 五、motif 应该围绕哪些 trigger 发现

如果直接在连续 L2 流上滑动所有窗口，会发现大量“普通盘口状态”，而不是 maker 策略。

更有效的是先围绕有信息含量的事件生成候选。

## Trigger A：主动成交冲击

例如：

[
\frac{\text{aggressive trade volume}}
{\text{触发前 best queue}}

> \theta
> ]

研究的问题：

> 流动性提供者在队列被吃掉后，是补、撤、移动，还是保持不动？

---

## Trigger B：最优队列被完全耗尽

观察：

* 原价是否快速恢复；
* 新 best 上是否立即补单；
* spread 是否扩大；
* 对侧报价是否同步移动；
* 中间价是否延续。

---

## Trigger C：大规模撤单

观察：

* 是否是单边撤退；
* 是否在价格跳变前发生；
* 是否在其他价位重新挂出；
* 对手侧是否同步补充。

---

## Trigger D：OFI 或 imbalance 冲击

观察 maker 是否：

* 顺势偏斜报价；
* 逆势吸收订单流；
* 减少危险一侧的流动性；
* 扩大报价距离。

---

## Trigger E：spread 扩大

观察：

* 谁先回到内档；
* 深度恢复速度；
* 新内档报价的持续时间；
* 是否快速被成交；
* 成交后价格是否反向。

---

# 六、真正要发现的不是“盘口 motif”，而是“条件响应 motif”

这是整个方法最重要的部分。

假设两个 episode 后半段看起来相同：

```text
ask 快速补单
spread 保持不变
```

但它们的前置状态完全不同：

* Episode A：买方 OFI 很强，ask 几乎被打穿；
* Episode B：市场平静，ask 只是普通更新。

不能把它们直接称为同一策略。

更合理的距离是：

[
d(i,j)
======

\lambda_S d_S(S_i^{pre},S_j^{pre})
+
\lambda_R d_R(R_i,R_j)
+
\lambda_O d_O(O_i,O_j)
]

其中最重要的是 (d_R)，但 (S^{pre}) 必须足够相似。

换句话说：

> **只在相似市场状态下，比较 maker 响应是否重复。**

---

# 七、先建立“正常市场响应”，再寻找异常稳定的剩余行为

一个常见错误是把交易所机械规则和全市场自然反应当成某个 maker 的策略。

例如：

```text
best ask 被吃掉
→ 新 best ask 出现
```

这可能只是订单簿机制，不是特殊策略。

因此建议先拟合一个条件基线：

[
\mathbb E[R_t\mid S_t]
]

可选模型：

* queue-reactive model；
* Hawkes process；
* 广义线性点过程；
* gradient boosting；
* 小型时序神经网络。

Queue-reactive 模型将不同价位队列看作状态，并令新增、撤单和成交的到达强度依赖当前盘口；这为估计被动单执行概率和正常队列演化提供了可解释基线。([arXiv][1])

然后计算响应残差：

[
R_i^{\mathrm{res}}
==================

R_i-
\mathbb E[R_i\mid S_i^{pre}]
]

motif discovery 主要在残差上做：

> 在控制了正常盘口机制以后，哪些额外的挂单/撤单响应仍然反复出现？

这比在原始深度上找 motif 更接近“策略”。

---

# 八、motif 搜索的具体做法

## 第一步：候选 episode 表征

每个 episode 生成：

[
X_i\in
\mathbb R^{T\times L\times C}
]

例如概念上：

```text
T：触发前后 150 个事件
L：买卖各 5 档
C：
  深度
  add flow
  cancel flow
  trade flow
  OFI
  spread
  mid move
  refill ratio
  queue persistence
```

---

## 第二步：降维

可以先用手工统计和 PCA：

* 触发前 imbalance；
* 冲击量占 best queue 比例；
* 1、5、10、20 个事件后的 refill ratio；
* cancellation asymmetry；
* 首次补单延迟；
* 最大深度恢复率；
* quote survival；
* 后续 mid-price return；
* post-fill markout。

也可以使用 CNN/LSTM 编码张量。DeepLOB 采用卷积提取订单簿价位结构，并使用 LSTM 处理时间依赖，说明“价位维度的空间结构 + 事件时间结构”是适合联合编码的形式。([arXiv][3])

第一版建议先使用解释性统计 + PCA，不要立刻上黑盒编码器。

---

## 第三步：寻找互为近邻的 episode

对每个 episode (i)，从不同日期或相隔足够远的时间中寻找最近邻：

[
j(i)
====

\arg\min_j d(z_i,z_j)
]

要求：

* 不来自同一局部波动事件；
* 最好来自不同交易日；
* 前置市场状态相似；
* 响应残差相似。

优先保留 mutual nearest neighbours：

[
j(i)=j,\qquad j(j)=i
]

Matrix Profile 的基本思想就是为每个子序列保存非重叠最近邻距离，并用低距离对定位反复出现的 motif；多变量版本可以将其扩展到多个 LOB 特征。([arXiv][4])

---

## 第四步：有限时间对齐

不同 episode 的反应速度可能略有不同：

```text
Episode A：5 个事件内完成补单
Episode B：8 个事件内完成补单
```

可以采用受限 DTW：

[
d_{\mathrm{DTW}}(X_i,X_j)
]

但要非常谨慎。

在 HFT 中，响应延迟本身可能就是策略核心。如果允许任意时间拉伸：

```text
50 微秒补单
和
5 毫秒补单
```

可能被错误认为是同一 motif。

因此建议：

* 形状比较允许小幅 event-time warp；
* 真实时钟延迟作为独立特征；
* DTW warping band 很窄；
* 不允许跨越关键市场事件；
* 对速度差异单独聚类。

DTW motif 方法适合发现形状相似但局部速度不同的子序列，但时间变形范围必须由领域约束限定。([arXiv][5])

---

## 第五步：建立 motif graph

每个 episode 是节点。

若两个 episode 同时满足：

[
d_S(S_i,S_j)<\tau_S
]

[
d_R(R_i^{res},R_j^{res})<\tau_R
]

[
d_O(O_i,O_j)<\tau_O
]

就在二者间连边。

图的社区对应 motif family。

例如可能得到：

### Motif 1：快速同价补充

```text
大单消耗 best
→ 极低延迟在原价补回
→ spread 不变
→ 后续价格延续较弱
```

### Motif 2：撤退并重新定价

```text
不平衡增大
→ 危险一侧大量撤单
→ 在远一档重新挂出
→ spread 暂时扩大
```

### Motif 3：单边深度偏斜

```text
连续买方订单流
→ bid 深度增加
→ ask 深度减少
→ 价格随后上移
```

### Motif 4：冲击吸收

```text
连续 aggressive flow
→ 被动侧不断补充
→ 价格不延续
→ 补单长期留在队列
```

这些是 maker-like response families，不应立即命名成具体公司的策略。

---

# 九、从 motif 上升到“策略”

单个 motif 只是固定情形下的一种重复响应。

真正的做市策略更像一个 policy：

[
A_t\sim\pi_k(A\mid S_t)
]

其中：

* (S_t)：盘口、订单流和波动状态；
* (A_t)：挂单、撤单、移动和数量选择；
* (k)：某种潜在策略或策略状态。

因此：

> **motif 是 policy 的局部切片。**

例如发现：

```text
市场平静时：双边补单
买方 OFI 高时：减少 ask、增加 bid
波动升高时：双边撤远
成交后：被成交一侧补单，另一侧降低深度
```

将这些 motif 组合起来，才可能形成一个稳定 maker policy。

---

# 十、适合建模策略族的模型

## 方案 1：Motif library + 规则图

建立：

[
\mathcal M=
{M_1,M_2,\ldots,M_K}
]

每个 motif 保存：

* 前置状态分布；
* 响应 prototype；
* 持续时间；
* 成交概率；
* 后续 markout；
* 出现频率；
* 市场状态条件。

然后分析 motif 之间的转移：

[
P(M_{t+1}=j\mid M_t=i,S_t)
]

解释性最好，适合 MVP。

---

## 方案 2：HMM/HSMM

定义隐藏策略状态：

[
Z_t\in
{
\text{neutral quoting},
\text{inventory skew},
\text{risk-off},
\text{queue defence},
\text{liquidity absorption}
}
]

观测是：

[
X_t=
[
\text{add},
\text{cancel},
\text{trade},
\text{depth},
\text{spread},
\text{markout}
]
]

HMM 学习：

[
P(Z_t\mid Z_{t-1})
]

和：

[
P(X_t\mid Z_t)
]

HSMM 还可以显式建模每种状态持续多久。

motif discovery 可以用来初始化隐藏状态 prototype，避免 HMM 从随机状态开始学出难以解释的分类。

---

## 方案 3：Switching policy / mixture of experts

假设存在若干潜在策略：

[
P(A_t\mid S_t)
==============

\sum_k
P(Z_t=k\mid S_t)
P(A_t\mid S_t,Z_t=k)
]

这里：

* gating model 决定当前使用哪种策略状态；
* 每个 expert 学习一种条件响应。

这比普通 motif 更接近真正的策略模型，但需要对 maker action 有较可靠的推断。

---

# 十一、如何判断某个 motif 真的是“稳定 maker 模式”

至少需要六个检验。

## 1. 跨交易日重现

不能只在同一天重复。

一个高置信 motif 应在不同日期、不同局部波动事件中出现。

## 2. 条件稳定

在相似的：

* spread；
* imbalance；
* volatility；
* trade intensity；
* queue depth；

条件下，响应仍然相似。

## 3. 超过基线模型

其残差响应应明显超出 queue-reactive/Hawkes 等正常市场动力学的预期。

## 4. 结果分布稳定

例如同一补单 motif 应有相对稳定的：

* fill probability；
* quote survival；
* spread capture proxy；
* post-fill markout；
* adverse-selection 风险。

做市收益评估必须显式处理成交概率和成交后的不利价格移动；忽略 adverse fills 或采用不现实的 fill 假设会明显高估策略表现。([arXiv][6])

## 5. 跨样本验证

在 discovery days 建 motif，在完全隔离的 validation days 搜索。

## 6. 与随机 surrogate 比较

例如：

* 打乱 trigger 与 response 的配对；
* 保留盘口状态但交换不同 episode 的后半段；
* 同一日内循环平移事件流；
* 保留事件强度但随机化买卖方向；
* 用基线点过程模拟数据。

如果模拟数据中也大量出现相同 motif，它可能只是市场结构的自然产物。

---

# 十二、L2 数据的识别极限

这点必须特别明确。

## 只有 aggregate L2/MBP

可以发现：

* 聚合流动性响应；
* 可能由一个或多个主要 maker 驱动的稳定模式；
* 订单簿中的“隐含做市状态”。

不能可靠知道：

* 是否由同一参与者产生；
* 某张单的真实队列位置；
* 撤单和新单是否属于同一 maker；
* 实际库存；
* hidden/iceberg 订单；
* 跨 venue 的整体策略；
* 某次撤单是主动策略还是撮合/数据修正。

## 有 MBO/order IDs

可以进一步重建：

* 单笔订单生命周期；
* cancel/replace；
* 队列位置；
* 补单链；
* 同一匿名订单源的行为一致性——前提是数据里确实有可关联标识。

## 有自身订单回报

如果研究的是自己的 maker 策略，则可以直接得到：

* action；
* queue position estimate；
* fills；
* inventory；
* realized spread；
* markout。

这时 motif 可以真正建立：

[
S_t\rightarrow A_t\rightarrow O_t
]

而不是从聚合盘口反推 (A_t)。

---

# 十三、推荐的 MVP

先选：

```text
单一标的
单一 venue
同一交易时段
前 5 档或前 10 档
20～60 个交易日
```

## Step 1：重建事件流

将每条消息归类为：

* bid add；
* bid cancel/net remove；
* ask add；
* ask cancel/net remove；
* aggressive buy；
* aggressive sell；
* price move。

## Step 2：只研究一个 trigger

第一版建议：

> best queue 被主动成交消耗超过触发前深度的 30%～50%。

不要一开始混合十种 trigger。

## Step 3：形成方向标准化 episode

```text
触发前 50 个事件
触发后 100 个事件
买卖方向镜像统一
价格变成相对 tick
数量除以局部深度尺度
```

## Step 4：构造 20～50 个解释性特征

重点包括：

* pre-trigger OFI；
* queue imbalance；
* impact fraction；
* refill latency；
* refill ratio；
* cancellation asymmetry；
* depth recovery curve；
* spread path；
* microprice path；
* quote survival；
* 10、50、100 个事件后的 mid-price markout。

## Step 5：建立基线响应模型

预测在给定前置状态下：

[
\hat R_i=
\mathbb E[R_i\mid S_i^{pre}]
]

motif 输入使用：

[
R_i-\hat R_i
]

## Step 6：PCA + 非邻近 kNN

* PCA 到 8～20 维；
* 最近邻必须来自不同时间段，最好不同交易日；
* 找 mutual nearest neighbours；
* 用原始响应曲线和受限 DTW 复核。

## Step 7：建立 motif graph

只连接：

* 前置状态相似；
* 残差响应相似；
* 时间上独立；
* outcome 分布相容；

的 episode。

## Step 8：生成 prototype

每个 motif 输出：

```text
平均前置盘口
平均深度响应曲线
平均 add/cancel/trade 流
refill latency 分布
refill ratio 分布
fill/quote survival
未来 markout 分布
跨日出现次数
```

## Step 9：冻结后验证

在未参与发现的交易日中搜索 motif，并检查其响应和 outcome 是否仍然稳定。

---

# 十四、最重要的结论

在 HFT L2+trades 场景中，motif 方法不能简单理解成：

> 找几段长得相似的订单簿图。

正确用法应该是：

[
\boxed{
\text{相似市场状态}
\rightarrow
\text{相似的异常流动性响应}
\rightarrow
\text{相似的执行与价格结果}
}
]

然后：

1. 对齐触发时间、相对价格、买卖方向和流动性尺度；
2. 用 queue-reactive/Hawkes 或统计模型扣除正常市场响应；
3. 在响应残差中寻找跨交易日重复的 motif pairs；
4. 将 motif 组合成状态依赖的 policy；
5. 用 fill、quote survival 和 post-fill markout 判断它是否具有稳定做市经济含义。

最终得到的不是：

> “我们发现了做市商 A。”

而更可能是：

> “在这类盘口状态下，市场中稳定存在一种低延迟补单、维持内档且后续 adverse selection 较低的流动性提供策略族。”

这才是仅凭 L2+trades 能支持的、可信的结论。

[1]: https://arxiv.org/pdf/1312.0563?utm_source=chatgpt.com "Simulating and analyzing order book data: The queue- ..."
[2]: https://arxiv.org/abs/1011.6402?utm_source=chatgpt.com "The Price Impact of Order Book Events"
[3]: https://arxiv.org/abs/1808.03668?utm_source=chatgpt.com "DeepLOB: Deep Convolutional Neural Networks for Limit Order Books"
[4]: https://arxiv.org/abs/1811.03064?utm_source=chatgpt.com "Towards a Near Universal Time Series Data Mining Tool: Introducing the Matrix Profile"
[5]: https://arxiv.org/abs/2009.07907?utm_source=chatgpt.com "Matrix Profile XXII: Exact Discovery of Time Series Motifs under DTW"
[6]: https://arxiv.org/html/2409.12721v2?ref=aligrithm.com&utm_source=chatgpt.com "Market Simulation under Adverse Selection"
