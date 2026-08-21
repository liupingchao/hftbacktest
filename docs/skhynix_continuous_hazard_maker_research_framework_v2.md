# SKHYNIX Continuous Conditional-Risk Maker Research Framework v2

Date: 2026-08-17

Revision: 2026-08-20

Status: user-approved active master research framework as of 2026-08-20.
Research Package Trust Kernel v1 was accepted on 2026-08-21. Stage H0-A is
unlocked but has not been dispatched. No v2 research build, collection,
private endpoint, order, cancel, deployment, or live authorization is granted
by this document.

Active sequencing:

```text
Research Package Trust Kernel v1 accepted
-> Stage H0-A support-only (unlocked, undispatched)
-> independent QA
-> Stage H0-B conditional-risk audit
-> independent QA
```

Accepted Trust Kernel pin:

```text
kernel_name = research_package_trust_kernel
kernel_version = v1
registry_revision = 1
registry_entry_sha256 = cae21d65bf447435bafc37508b8ca00643a0742b37e0f404148cab92818c90c9
kernel_source_tree_sha256 = cee2395afad9420c38235ba195bf030e92330015e1a15937ebc22fa707c80203
acceptance_task_id = 0820T001
```

前置文档：

- v1 研究方案（已冻结，不因本文件修改）：
  `docs/skhynix_trigger_aligned_episode_research_implementation_plan.md`
- Stage 4 执行与返修复盘：
  `docs/skhynix_stage4_episode_v3_execution_repair_postmortem.md`
- 双交易所理论框架：
  `docs/binance_hyperliquid_slice_v2.md`
  `docs/binance_hyperliquid_slice_v2_appendix.md`

## 1. 文档定位与 v1 的关系

本文件是一份**新命名的研究框架合同**，不是对 v1 的静默修订。
为保留已有引用与 Git 历史，文件名继续使用 `continuous_hazard`；正文中的
正式统计对象统一为 `conditional adverse-event risk`。

约束关系：

1. v1 contract、Stage 1–4 已验收 packages、frozen trigger contract 全部
   保持不可变。本框架只**消费** accepted artifacts，不重开、不重建。
2. Episode v3（Family A/B）在 v2 中降级为连续过程的一种 landmark view，
   仍是 auditable dependency，不删除。
3. 用户已于 `2026-08-20` 批准本框架。v1 Ordered Research Queue 中尚未
   派发的 Stage 5 及之后各项从该日起**停止按 v1 编号推进**；其中仍然
   有效的问题（如 cross-spread vs queue-shock 增量信息）以移植后的形式
   进入本框架。
4. frozen queue-shock detector 的任何阈值不因 v2 改变。detector 在 v2 中
   的角色改变（见第 3 节），但其定义、代码与 parity 证据原样沿用。

## 2. 框架转向的事实依据

v1 的 episode 框架隐含前提：trigger 是**稀有的例外事件**，市场大部分时间
处于安静基线，maker 在基线中挂单，仅在事件时刻决策 KEEP/CANCEL。

Stage 2（0815T001，已验收）的密度证据将该前提证伪：

| 事实 | Jul30 | Aug03 | Aug04 |
| --- | ---: | ---: | ---: |
| Family A candidate rate（每秒） | 18.65 | 7.09 | 9.37 |
| inter-trigger p50 / p99（ms，all candidates） | 29.1 / 344.7 | — | — |
| 2000ms 窗口 union 覆盖率 | 0.9998 | 0.9924 | 0.9999 |
| 2000ms overlap block 数 | 9 | 232 | 6 |
| 最长连续触发 run | ≈30min（整个 segment） | ≈30min | ≈1.8h |

并且 Stage 2 的 named sensitivity 证明：**收紧触发不制造统计独立性**。
从 `impact_ge_070`（保留 90.5%）到 `same_side_refractory_500ms`（保留
16.4%）到 `first_per_primary_flow_episode`（保留 3.9%，10,536 flows），
Jul30 的 2000ms overlap block 数始终为 `9`。窗口链由时间连接，不由样本
选择连接。

结论：

```text
对 SKHYNIX，trigger 所定义的 shock-dose 状态不是稀有例外，
而是一条近连续的可观测状态过程。
Stage 2 证明的是 detector 输出密集和 episode 窗口不独立，
不单独证明目标 venue 的 adverse-event risk 高。
该连续状态过程本身是 evidence；风险研究框架应当由它出发设计，
而不是通过收紧阈值强迫数据回到稀有事件假设。
```

由该事实推出的三个风险研究结论（本框架的设计公理）：

1. **姿态先于反应。** 当 detector candidates 每秒出现约 7–19 次，
   per-episode 的
   KEEP/CANCEL overlay 退化：要么永远在撤单（零 spread capture），要么
   阈值高到形同虚设。若后续风险与机会两侧证据都支持，真正的决策变量
   应是 maker 的连续姿态：
   是否在场、报价距离、报价数量，作为连续 shock-dose/risk 状态的函数。
   KEEP/CANCEL 是该连续策略退化为两档的特例。
2. **反应式保护只在 risk 可预测地时变时有价值。** 若 risk 基本平坦，
   当前可观测状态不支持反应式风险信号；更宽、更小或不做只能列为静态
   posture candidates，最终选择仍需机会侧与价值侧证据。
   "risk 时变性与可预测性"因此是反应式风险信号是否存在的 go/no-go
   问题；它本身不决定最优报价姿态。
3. **策略默认极性可能反转，但不是本阶段输出。**
   quote-by-default / cancel-on-danger 的失误
   是 adverse fill；flat-by-default / quote-on-safety 的失误是错过
   capture。若后续证据证明高 risk 为常态，前者尾部可能远重于后者。
   默认极性是
   后续联合 risk/opportunity 研究的输出，不是本框架预设，也不能由
   public adverse-risk 单侧证据推出。

## 3. 研究对象重定义

### 3.1 从 Palm distribution 到条件风险

v1 估计的是围绕离散 trigger 的 event-conditioned 分布：

\[
P\left(R_i,\ O_i^{market}\mid S_i^{pre},\ T_i\right)
\quad\text{around discrete }\tau_i
\]

v2 估计整条 common L2 timeline 上的**条件 adverse-event risk**。对目标 venue
（Hyperliquid）的方向侧 \(s\in\{bid,\ ask\}\)、参考报价距离 \(\delta\)、
预测 horizon \(h\)：

\[
p_{s,\delta,h}(t)
=
P\left(
\text{side-}s\ \text{adverse event in }(t,\ t+h]
\mid
\mathcal F_t^{obs}
\right)
\]

其中 \(\mathcal F_t^{obs}\) 是服务器在 \(t\) 时刻已接收信息的 filtration，
沿用 v1 的 strict-as-of、observed-at ledger 与 no-look-ahead 纪律。

\(p_{s,\delta,h}(t)\) 是固定 horizon 的条件风险或 cumulative incidence，
不是瞬时 hazard intensity。若后续发布离散时间 hazard，则必须另行定义
at-risk set、事件 onset/reset、exposure interval 与 recurrent-event 规则；
不得将固定 horizon 概率直接命名为 \(\lambda(t)\)。

### 3.2 Trigger 的新角色

frozen queue-shock detector 在 v2 中承担且仅承担两个角色：

1. **状态特征。** trailing shock dose（近期冲击强度、方向、剂量路径）
   是 \(\mathcal F_t^{obs}\) 中的一组输入特征，与 cross-spread 特征并列。
2. **验证锚点。** 已验收的 candidate/confirmed 时刻集合用于检查 risk
   模型在已知密集时段附近的校准，以及与 Episode v3 view 的交叉核对。

trigger 不再是抽样原点。任何以 trigger 为原点的 per-event 统计只作为
诊断视图存在。

### 3.3 与理论框架文档的关系

`binance_hyperliquid_slice_v2*.md` 的核心命题——用可观测信息集定义
stopping time、response prefix 更新未来分布、动作反事实价值——在 v2 中
全部保留。改变的只是：随机时钟从"稀有事件时刻"退化为"连续时间本身"，
因为数据表明 frozen detector 的 candidate-intensity process 几乎处处为正。

## 4. 主研究问题（按 go/no-go 顺序）

### RQ1：adverse-event risk 时变性（go/no-go）

短 horizon adverse-event risk 在 calendar time 上的起伏有多大？

产出：

- 无条件 adverse-event rate 随时间的路径（按 session/segment）；
- 固定时间块的 excess dispersion、状态持续期与变点诊断；
- dependence-preserving stationary null 下的时变性检验。

若时变性不足（gate 见第 11 节），研究结论为
`quote_risk_flat_reactive_signal_not_indicated`，反应式风险建模停止。

### RQ2：可预测性与增量信息（v1 B1-vs-B3 的移植）

risk 的起伏能否从可观测状态提前读出？queue-shock dose 在
cross-spread 状态之外是否有稳定增量？

产出包括粗状态分箱的经验 conditional-risk 谱、严格 out-of-fold 预测
分层、H0–H4 校准与 loss comparison。

预注册嵌套特征集（移植 v1 §12）：

```text
H0 = direction/side + session/time context
H1 = H0 + cross-spread level/change/age/residual
H2 = H1 + full dual-venue book/flow state
H3 = H2 + trailing queue-shock dose (frozen detector outputs)
H4 = H3 + Hyperliquid short-horizon own-venue prefix
```

primary dose 增量比较是 **H3 vs H2**：在完整 dual-venue book/flow
状态之外，queue-shock dose 是否仍提供稳定增量。H3 vs H1 仅作为
"全部附加状态相对 cross-spread"的 secondary decomposition，不得将其
改善归因于 dose。

### RQ3：regime residual dwell vs 延迟预算（风险信号可执行性）

风险信号能利用的不是单次事件的提前量，而是"状态在首次可识别之后，
扣除反应延迟仍继续存在"。

定义 out-of-fold 预测 risk 高于冻结阈值 \(q\) 的时段为 high-risk
regime，测量：

- high/low regime 的 dwell time 分布（p10/p50/p90）；
- regime 切换率；
- 与假设延迟档位 `25/50/100/250/500ms` 的对比（沿用 v1 Gate E 档位）；
- 冻结 threshold / hysteresis / debounce 后的首次可识别时刻；
- 从首次可识别时刻扣除延迟后的 residual dwell 分布；
- 进入/退出 regime 时刻的 risk 路径形状（切换是否可提前观测）。

已有的先验证据：Jul30 cluster 时长 p50 ≈ 80ms、flow 时长
p50 ≈ 477ms / p90 ≈ 2.5s——flow 尺度的持续期显著长于典型撤单延迟，
说明该问题值得测量而非先验否定。

三问的 go/no-go 链：

```text
RQ1 fail -> 反应式风险信号不成立，停止
RQ1 pass, RQ2 fail -> 时变但不可预测，停止或转向数据扩充
RQ1+RQ2 pass, RQ3 fail -> 可预测但延迟不可执行，转向更低频风险问题
RQ1+RQ2+RQ3 pass -> 风险信号研究成立；机会侧研究通过后才可进入姿态设计
```

### RQ4：maker opportunity companion boundary（后续独立合同）

本框架只回答 public quote-risk 信号是否存在，不回答 maker 净价值。
任何 posture、quote-by-default / flat-by-default、报价距离或"不适合做
maker"结论，都必须由后续独立合同同时估计：

- `quote_contact` / potential spread-capture opportunity；
- contact 后的方向归一化 markout；
- move-through 与 quote survival；
- 在明确 fee/rebate、queue/fill 假设层级下的 value bound。

RQ4 不属于 Stage H0，也不因 RQ1–RQ3 通过而自动通过。

## 5. 状态向量合同

完整继承 v1 §8.1 的特征族与纪律，不新增私有信息：

- cross-venue：`d_bh/d_hb/risk_gap`、basis residual、gap 变化率、
  gap 归因分解、双边 feed age、connection epoch、degraded mask；
- Binance：BBO/spread、best queue、top-5 depth/imbalance、signed flow、
  OFI、replenishment/depletion、波动与更新强度；
- Binance trailing shock dose（第 3.2 节角色 1）：最近 `w` 窗口内
  frozen detector 的 candidate/confirmed 计数、方向、impact ratio 路径、
  cluster/flow 成员状态；
- Hyperliquid：BBO/spread、impacted/opposite quantity、fast-L2 depth、
  事件强度、方向性移动、staleness / no-new-information 状态；
- context：session、time-of-session、underlying（KRX）open/close/auction
  状态、波动/流动性 bucket。

纪律沿用：每个特征行携带
`value / observed_at_ns / source_event_id / source_book_version /
calculation_version / availability_reason`；
`observed_at <= t` 为全量断言而非抽查。

own inventory、own order、queue position、fee、fill 字段不存在于数据中，
禁止臆造。

## 6. Quote contact、adverse event 与 outcome 合同

### 6.1 参考报价

对侧 \(s\)、距离 \(\delta\in\{0,\ 1\ \text{tick}\}\)
定义假设参考报价位。side-aware 价格方向必须冻结为：

```text
bid quote: best_bid(t) - delta
ask quote: best_ask(t) + delta
```

参考报价使用 \(t\) 时刻 strict-as-of 的目标 venue BBO 与该 session
冻结的 tick-size contract。`delta=0` 为 at-best，`delta=1` 为离市场
一档，不使用含混的 `best±1` 表述。

### 6.2 Outcome taxonomy

public evidence 分为四类，不得合并命名：

```text
quote_contact:
  public_trade_reaches_quote

adverse_transition:
  public_bbo_moves_through_quote

post_contact_adverse:
  contact_followed_by_adverse_markout
  maximum_adverse_excursion_after_contact

survival:
  public_quote_survives_horizon
```

`public_trade_reaches_quote` 只表示公开市场触达假设报价，不表示真实 fill，
也不单独表示 adverse selection。v2 的默认 primary target 冻结为
`public_bbo_moves_through_quote`，默认 primary distance 为 `delta=0`。
`contact_followed_by_adverse_markout` 是 secondary target，必须先冻结
markout horizon、方向归一化与 interval semantics 才可评分。quote contact
作为 opportunity diagnostic 单独报告，不得与 adverse target 做 OR-union。

命名纪律沿用：字段名不得含 `fill/filled/execution_pnl/own_order`。

### 6.3 Horizon 与采样

预测 horizon：

```text
h ∈ {50, 100, 250, 500} ms
```

1000/2000ms 只作为 descriptive atlas，不进入 primary gates（依据：
2000ms 尺度独立块数 Jul30=9、Aug04=6，不支持校准声明）。

primary evaluation grid 是与绝对 receive-time 对齐的固定 `10ms`
calendar-time grid，不是 message-arrival grid。每个合格 grid interval
贡献相同的 `10ms` exposure：

- 特征只使用 grid endpoint 时 strict-as-of 可见的状态；
- forward-filled state 必须携带 source timestamp、source age、
  connection epoch、quality mask 与 no-new-information flag；
- 无新消息的 interval 仍表示真实 calendar-time exposure，但不增加独立
  信息量，不得按行解释为 \(N_{eff}\)；
- event-arrival grid 只作 market-activity robustness view，不得替代
  primary calendar-time 结果；
- segment、epoch、quality gap 与 unavailable state 不产生 exposure，
  不得以 forward fill 跨越。

primary horizon 只能由 Stage H0-A 的 support-only 规则选择：按
`50 -> 100 -> 250 -> 500ms` 顺序，选择第一个在至少两个 session 中同时满足
下列默认条件的 horizon：

```text
quality-eligible calendar exposure >= 95%
fully identified binary endpoint fraction >= 90%
interval-likelihood eligible fraction >= 95%
complete 60s calendar blocks >= 20 per qualifying session
```

该选择只允许读取 coverage、cadence、censoring 与 dependence metadata，
不得读取 adverse rate、feature-conditioned rate、loss 或 effect size。
选择结果、输入 inventory 与代码 SHA 必须在 H0-B 打开 outcome aggregate
前写入并 fsync 一个 immutable horizon-freeze manifest。

### 6.4 Censoring

完整继承 Stage 4 已验收的 interval/right/segment/quality censoring 机制。
Hyperliquid timing 保持 interval censoring，禁止点化；horizon 端点落入
censored 区间的标签显式标记为 interval-ambiguous，不得静默取边界。

评分合同：

- interval/right-censored event time 的 primary score 是沿用 v1 §11.4
  的 interval log loss；
- binary Brier/log loss 只在 horizon endpoint 可识别为 `0/1` 的 grid
  intervals 上计算，并同时发布 identified fraction；
- interval-ambiguous rows 保留在 interval likelihood 与上下界
  sensitivity 中，不得为方便计算而删除、点化或取边界；
- 若 primary horizon 未达到第 6.3 节 identification/coverage gate，
  结果只能是 `inconclusive_data_quality_or_coverage`。

## 7. 估计器合同

按研究问题与复杂度升序：

1. **calendar-block rate / dispersion estimator**：固定 `60s` block 为
   RQ1 primary；`10/30/120s` 只作 robustness；
2. **经验分箱 conditional risk**（cross-spread bins × dose bins × side）：
   可完全审计，是 RQ2 的 baseline；
3. **正则化 logistic / discrete-time event-risk regression**：
   H0–H4 嵌套消融的 RQ2 primary estimator；
4. **quantile / 非线性模型**：仅作 robustness，不得成为唯一支持证据。

神经网络与深度序列模型在本框架 v2 内不使用。

case-retrieval（v1 §11）在连续框架下不再是 primary estimator；若后续
恢复使用，需单独命名合同。

## 8. 评估与统计单元

- **within-session**：严格 expanding/rolling walk-forward，训练块必须
  早于测试块；train/test 边界使用至少 `max(primary_horizon, 500ms)` 的
  purge + embargo，禁止随机切分；
- **out-of-fold**：bin edges、risk threshold、scaler、regularization、
  calibration 与十分位边界只在过去训练块拟合，测试块不得回流；
- **不确定性**：flow-block / time-block bootstrap，禁止行级 IID bootstrap；
- **复制单元**：session。跨 session 声明的证据强度被独立采集日数封顶，
  报告必须显式给出该封顶；
- **评分**：event-time primary 用 interval log loss；fully identified
  binary view 用 Brier/log loss + reliability table；分层单调性只使用
  out-of-fold 预测十分位；dwell time 使用 interval-aware 分布分位数；
- **primary multiplicity**：只有冻结的
  `target × delta × horizon × latency × side_aggregation` tuple 可进入
  gates；默认 side aggregation 为 bid/ask session score 等权平均，单侧及
  其他 horizon/distance/latency 结果均标为 secondary，不得择优晋升；
- **session aggregation**：先在 session 内评分，再对 qualifying sessions
  等权汇总；pooled grid rows 不得替代 session-level 结果；
- **降级发布**（继承 Stage 2 纪律）：所有结构计数标注
  `never interpreted as universal N_eff`。

evidence label 沿用 v1 §4 词表，新增：

```text
prospective_monitoring
prospective_consumed_validation
prospective_final_holdout
```

不再使用无消费状态的通用 `prospective_holdout` 标签。

## 9. 资产继承与降级表

| v1 / Stage 1–4 资产 | v2 中的地位 |
| --- | --- |
| common L2 timeline（R0/R1） | 主数据平面，直接复用 |
| frozen trigger contract + parity（Stage 3） | dose 特征生成器 + 验证锚点，阈值不变 |
| Stage 2 density/merging/sensitivity package | sampling/dependence prior + trigger-dose cross-check；不直接证明 adverse risk |
| Stage 4 Episode v3 Family A/B（1.5GB package） | landmark view + 交叉核对源；不重开 |
| observed-at feature ledger 机制 | 全量继承 |
| interval/right/quality/segment censoring 机制 | 全量继承 |
| v1 B0–B4 嵌套问题 | 移植为 H0–H4 |
| v1 Gate E 延迟档位 | 移植为 RQ3 延迟预算档位 |
| per-episode Palm 统计 | 降级为诊断视图 |
| case retrieval as primary estimator | 移出 primary，恢复需新合同 |
| v1 Ordered Queue Stage 5+ | 停止按 v1 编号推进 |

## 10. Stage H0：conditional-risk 时变性审计（首个可派发单元）

v2 的第一个执行单元是一个**只读、小输出**的审计任务，消费已验收
packages，不新建大规模数据面。它同时是 trust kernel（见第 13 节）的
首个试点消费者。

执行 stage `Stage H0` 与第 4 节 feature set `H0` 是两个不同标识；实现、
manifest 与报告中必须分别序列化为 `stage_h0` 和 `feature_set_h0`。

输入（只读）：

- Jul30 / Aug03 / Aug04 accepted R0/R1 timeline；
- Stage 2 density/merging/sensitivity package；
- Stage 4 Episode v3 package（仅交叉核对）；
- Aug07 event rows 在本 stage **不打开**。

H0-A 产出：

1. `10ms` calendar-grid coverage、source cadence、censoring 与 endpoint
   identification 表；
2. 各 horizon × session 的依赖块数与可行性表（50/100/250/500ms
   primary candidates；1000/2000ms descriptive）；
3. 不含 outcome value/effect 的 immutable primary-tuple freeze manifest。

H0-B 产出：

1. 各 session、各 side、冻结 horizon 的无条件 adverse-event rate 路径与
   `60s` block dispersion；
2. 粗状态分箱（cross-spread × dose）的经验 conditional-risk 表；
3. out-of-fold 预测 risk 分层的 realized-rate spread
   （RQ2 先导诊断，不属于 RQ1 primary gate）；
4. high/low risk regime 的 dwell 与 residual dwell 分布 vs 冻结延迟
   （RQ3 初值）；
5. H0 vs H1 的粗校准差（RQ2 先导信号，仅使用冻结的 coarse estimator）。

明确非目标：不拟合正式多变量模型、不做 estimator/model 选择、不触碰
Aug07、不改任何已验收 artifact。

Stage H0 分成两个不可倒置的 envelope：

```text
H0-A support-only
  只发布 grid coverage、cadence、censoring、identification 和 dependence；
  按 §6.3 机械选择并冻结 primary horizon；
  不发布 adverse rate、conditioned rate、loss 或 effect size。

H0-B conditional-risk audit
  只消费 H0-A 冻结的 primary tuple；
  发布 RQ1 block variation、RQ2 coarse out-of-fold diagnostics 与
  RQ3 residual-dwell feasibility；
  不得因 H0-B 结果更改 primary horizon 或 target。
```

H0-B 的结果决定 v2 主建模阶段是否派发；primary horizon 只能由 H0-A 的
support-only 规则决定。

## 11. 预注册 gates 与决策出口

以下数值在 v2 contract 冻结时最终确定；本稿给出默认值。

### Gate H-A：时变性（RQ1）

- primary `60s` calendar block 的 adverse rate dispersion 在 ≥2 个
  session 中超过 dependence-preserving stationary null 的 `95%` 分位；
- block variation 不能由 segment/epoch/quality/cadence 边界解释；
- 预测十分位 realized-rate spread 不属于本 gate，只进入 Gate H-B；
- primary 不确定性以 time-block bootstrap 报告，flow-block bootstrap
  作为 robustness。

### Gate H-B：可预测性与增量（RQ2）

- primary normalized loss 定义为
  `session_loss(model) / session_loss(reference)`，先按 session 计算；
- H1/H0 与 H3/H2 的默认 gate 均要求 normalized interval log loss
  `<=0.99`，在 ≥2 个 session 保持，且无 session `>1.00`；
- time/flow-block bootstrap 的 `90%` CI 上界必须 `<1.00`；
- identified binary Brier/log loss 与 reliability table 必须方向一致；
- 改善不得来自单一事后状态桶；
- H3 vs H1 只作 secondary decomposition；
- top-decile / bottom-decile realized-rate ratio 与单调性只作
  calibration diagnostic，不替代 loss gate。

### Gate H-C：可执行性（RQ3）

- gate-relevant latency 在 outcome 打开前冻结，默认 `100ms`；其余
  `25/50/250/500ms` 只作 sensitivity；
- risk threshold、hysteresis 与 debounce 只用过去训练块拟合；
- \(t_{detect}\) 是测试块中首次满足冻结 entry rule 的 grid endpoint，
  \(t_{exit}\) 是首次满足冻结 exit rule 的 endpoint；
- residual dwell interval 定义为
  `t_exit - (t_detect + frozen_latency)`，total dwell 仅作描述；
- interval-aware residual-dwell p50 的下界在 ≥2 个 session 中必须
  `>0`，且 identified fraction 默认 `>=90%`；
- regime entry 必须可由 \(t_{detect}\) 时 strict-as-of 特征重放；
- 结果按 side/session/underlying-regime 分层报告。

### 决策出口（唯一 primary classification）

```text
conditional_quote_risk_signal_supported
  RQ1/RQ2/RQ3 全过；仅证明 public quote-risk 信号可研究。
  进入姿态策略设计前仍需 RQ4 opportunity companion contract。

quote_risk_flat_reactive_signal_not_indicated
  时变性不足；不支持反应式风险信号，不推出更宽/更小/不做。

quote_risk_variation_unpredictable
  时变但状态不可预测；转向数据扩充或特征扩充，不做策略声明。

predictable_but_not_latency_actionable
  可预测但 residual dwell/延迟不匹配；记录并限定于更低频风险问题。

cross_spread_supported_dose_increment_not_supported
  H1 成立而 H3 相对 H2 无稳定增量；queue-shock dose 降级为纯锚点。

cross_session_unstable_needs_more_sessions
  within-session 成立但跨 session 不稳；触发第 12 节数据扩充后重估。

inconclusive_data_quality_or_coverage
  支撑、覆盖或数据质量不足。
```

任何出口都不授权订单、私有 endpoint 或 live 部署。
任何出口都不单独证明 maker profitability、最优姿态或该标的是否适合做
maker。

## 12. 数据扩充：滚动采集与 prospective consumption lifecycle

连续框架解决概念错配，不解决独立日数不足。跨 session 校准声明的证据
强度由独立采集日数封顶（当前为 4）。因此：

1. v1 §20 的 "does not collect new data" non-goal 在 v2 中**显式撤销**，
   替换为一个常态化滚动采集组件；
2. 采集使用已验收的 collection + QA 管线，目标节奏示例：每周 3–4 个
   session、覆盖不同 time-of-session 与 underlying（KRX）regime，
   持续至少数周；
3. 新 session 在 outcome rows 第一次打开前，必须在 immutable
   consumption ledger 中被指定为
   `prospective_monitoring`、`prospective_consumed_validation` 或
   `prospective_final_holdout`；仅仅晚于 contract freeze 采集不自动构成
   holdout；
4. monitoring session 可用于漂移报警或下一版本设计，一旦查看 outcome
   并据此修改当前版本，即永久转为 consumed validation；
5. `cross_session_unstable_needs_more_sessions` 的当前版本重估只能消费
   `prospective_consumed_validation`，不得回收 discovery sessions，也
   不得把已消费 session 重新标成 final holdout；
6. `prospective_final_holdout` 只允许在 model、primary tuple、threshold、
   calibration 与解释全部冻结后执行一次；首次打开记录、代码 SHA、输入
   inventory 和失败尝试必须 durable 保留；
7. final holdout 结果不得用于同一版本 refit。任何因其结果产生的修改必须
   创建新版本，并把该 session 标为前一版本的 consumed validation。

## 13. 执行与验收纪律

完整继承复盘 §14/§16 的方法修订，并作为硬性前置：

1. **Trust kernel 前置。** 方向 1 的
   admission/attestation 共享模块（exact lstat tree universe、
   source-owned aggregate evidence contract、durable evidence gate、
   atomic publication、fail-closed admission CLI）先于任何 v2 数据面
   构建完成并通过一次独立 QA；此后每个 v2 stage 只 pin accepted kernel
   版本，stage QA 不再重新审理信任边界本身；
2. **Package identity 分层**：
   `research_data / runtime_contract / publication_envelope`；
   trust-only repair 不重建数据层；
3. **load-bearing surface matrix** 在每个 stage 派发文档中内嵌并冻结；
   任一 output family 无对应 negative mutation test 即不得进入 `待验收`；
4. **hostile preflight 先于 full build**：全部 negative 场景先在秒级
   fixture 上失败关闭，才允许第一次全量构建；
5. **复盘 §16 七条 exit criteria** 逐条适用；QA 在任一条未定义时直接
   拒绝，即使全部 positive tests 通过；
6. 每个 stage 的规模纪律：优先小输出（Stage H0 级别）验证方法，再扩大
   数据面；禁止在未通过 H0 gates 前构建 GB 级 v2 数据面。

## 14. Non-goals

本框架 v2 不：

- 修改或重建任何已验收 v1/Stage 1–4 artifact；
- 修改 frozen trigger detector 的任何阈值；
- 访问私有/账户/order/cancel endpoint，或提交/取消订单；
- 推断真实 queue position、fill probability、fee、inventory、PnL；
- 估计 KEEP/CANCEL 或任何动作的因果 EV；
- 仅凭 adverse-risk 单侧证据选择 quote-by-default / flat-by-default；
- 仅凭 RQ1–RQ3 判定该标的适合或不适合做 maker；
- 优化 GLFT 参数或任何报价参数；
- 部署或授权 live 策略；
- 在 Stage H0 阶段打开 Aug07 event rows；
- 将结构行数解释为统计样本量。

## 15. 完成定义

v2 研究完成当且仅当 accepted packages 回答：

1. SKHYNIX 目标 venue 的短 horizon adverse-event risk 是否显著时变？
2. 该时变是否可由决策时点可观测状态预测？queue-shock dose 是否在
   cross-spread 之外提供稳定增量？
3. 高危状态在首次可识别并扣除现实延迟预算后，是否仍有正 residual
   dwell，使风险信号在原则上可执行？
4. 上述结论是否在 ≥2 个 session 上稳定，并在一次性
   `prospective_final_holdout` 上保持？
5. 由此得出的风险研究结论是什么：条件风险信号成立、风险时变但不可
   预测、可预测但延迟不可执行，或反应式风险信号不成立？

在第 11 节某一出口被正式接受之前，本项目可以发布 auditable 的 risk
审计与 descriptive atlas，但不得声称任何可部署的 maker 规则。

## 16. 相对 v1 的变更记录（changelog）

| 变更 | v1 | v2 | 依据 |
| --- | --- | --- | --- |
| 统计框架 | trigger-aligned episode / Palm | 连续 conditional adverse-event risk | Stage 2 密度证据（§2） |
| trigger 角色 | 抽样原点 + 特征 | dose 特征 + 验证锚点 | 同上 |
| primary horizon | 至 2000ms | H0-A support-only 机械选择 ≤500ms；1000/2000ms descriptive | 防止 outcome-driven horizon selection |
| 时间采样 | trigger/event-time grid | 固定 10ms calendar exposure grid | 防止 message-rate weighting |
| primary estimator | case retrieval | RQ1 block dispersion；RQ2 分箱 baseline + 正则化回归 | 分离时变性与可预测性 |
| censoring score | mixed binary/event-time | interval log loss primary；Brier 仅 identified view | 不删除 interval-ambiguous rows |
| 增量问题 | B0–B4 | H0–H4（移植） | 保留核心科学问题 |
| actionability | per-episode margin | residual regime dwell vs 延迟预算 | 连续过程无单事件提前量 |
| posture 结论 | quote protection candidate | 风险信号出口；posture 需 RQ4 | public risk 单侧证据不识别 maker EV |
| 新数据 | non-goal | 滚动采集 + immutable consumption lifecycle | holdout 一经查看即被消费 |
| 验收架构 | 属性列表 | trust kernel + surface matrix + preflight | Stage 4 复盘 §11/§14/§16 |
| Gate 度量 | "5% worse" 未定义 | 冻结时点名评分与 normalization | 复盘教训 |
