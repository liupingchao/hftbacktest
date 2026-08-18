# SKHYNIX Continuous Hazard Maker Research Framework v2

Date: 2026-08-17

Status: draft for user review; not dispatched; no research build, collection,
private endpoint, order, cancel, deployment, or live authorization is granted
by this document

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

约束关系：

1. v1 contract、Stage 1–4 已验收 packages、frozen trigger contract 全部
   保持不可变。本框架只**消费** accepted artifacts，不重开、不重建。
2. Episode v3（Family A/B）在 v2 中降级为连续过程的一种 landmark view，
   仍是 auditable dependency，不删除。
3. v1 Ordered Research Queue 中尚未派发的 Stage 5 及之后各项，在 v2 获得
   用户批准后**停止按 v1 编号推进**；其中仍然有效的问题（如 cross-spread
   vs queue-shock 增量信息）以移植后的形式进入本框架。
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
对 SKHYNIX，trigger 所定义的"危险状态"不是例外，而是环境本身。
被设计为 filter 的 trigger，实际测量到的是一条近连续的危险强度过程。
该连续过程本身是 evidence；策略框架应当由它出发设计，
而不是通过收紧阈值强迫数据回到稀有事件假设。
```

由该事实推出的三个策略层结论（本框架的设计公理）：

1. **姿态先于反应。** 当危险信号每秒出现约 7–19 次，per-episode 的
   KEEP/CANCEL overlay 退化：要么永远在撤单（零 spread capture），要么
   阈值高到形同虚设。真正的决策变量是 maker 的连续姿态：
   是否在场、报价距离、报价数量，作为连续危险状态的函数。
   KEEP/CANCEL 是该连续策略退化为两档的特例。
2. **反应式保护只在 hazard 可预测地时变时有价值。** 若危险度恒定地高，
   正确答案是静态的（更宽、更小、或不做），不需要信号系统。
   "hazard 时变性与可预测性"因此是整个策略是否存在的 go/no-go 问题，
   且"hazard 基本平坦，不适合反应式 maker"是一个有效结论。
3. **策略默认极性可能反转。** quote-by-default / cancel-on-danger 的失误
   是 adverse fill；flat-by-default / quote-on-safety 的失误是错过
   capture。在高危为常态的环境中，前者尾部远重于后者。默认极性是
   研究输出，不是预设。

## 3. 研究对象重定义

### 3.1 从 Palm distribution 到条件强度

v1 估计的是围绕离散 trigger 的 event-conditioned 分布：

\[
P\left(R_i,\ O_i^{market}\mid S_i^{pre},\ T_i\right)
\quad\text{around discrete }\tau_i
\]

v2 估计整条 common L2 timeline 上的**条件危险强度**。对目标 venue
（Hyperliquid）的方向侧 \(s\in\{bid,\ ask\}\)、参考报价距离 \(\delta\)、
预测 horizon \(h\)：

\[
\lambda_{s,\delta,h}(t)
=
P\left(
\text{side-}s\ \text{adverse event in }(t,\ t+h]
\mid
\mathcal F_t^{obs}
\right)
\]

其中 \(\mathcal F_t^{obs}\) 是服务器在 \(t\) 时刻已接收信息的 filtration，
沿用 v1 的 strict-as-of、observed-at ledger 与 no-look-ahead 纪律。

### 3.2 Trigger 的新角色

frozen queue-shock detector 在 v2 中承担且仅承担两个角色：

1. **状态特征。** trailing shock dose（近期冲击强度、方向、剂量路径）
   是 \(\mathcal F_t^{obs}\) 中的一组输入特征，与 cross-spread 特征并列。
2. **验证锚点。** 已验收的 candidate/confirmed 时刻集合用于检查 hazard
   模型在已知密集时段附近的校准，以及与 Episode v3 view 的交叉核对。

trigger 不再是抽样原点。任何以 trigger 为原点的 per-event 统计只作为
诊断视图存在。

### 3.3 与理论框架文档的关系

`binance_hyperliquid_slice_v2*.md` 的核心命题——用可观测信息集定义
stopping time、response prefix 更新未来分布、动作反事实价值——在 v2 中
全部保留。改变的只是：随机时钟从"稀有事件时刻"退化为"连续时间本身"，
因为数据表明事件强度过程几乎处处为正。

## 4. 主研究问题（按 go/no-go 顺序）

### RQ1：hazard 时变性（go/no-go）

短 horizon hazard 在时间上的起伏有多大？

产出：

- 无条件 adverse-event rate 随时间的路径（按 session/segment）；
- 粗状态分箱下的经验 hazard 谱；
- 预测 hazard 分布的高低分位比（primary：p90/p10）；
- 按预测 hazard 十分位分层的 realized outcome rate 单调性与 spread。

若时变性不足（gate 见第 11 节），研究结论为
`hazard_flat_static_posture_indicated`，后续建模停止。

### RQ2：可预测性与增量信息（v1 B1-vs-B3 的移植）

hazard 的起伏能否从可观测状态提前读出？queue-shock dose 在
cross-spread 状态之外是否有稳定增量？

预注册嵌套特征集（移植 v1 §12）：

```text
H0 = direction/side + session/time context
H1 = H0 + cross-spread level/change/age/residual
H2 = H1 + full dual-venue book/flow state
H3 = H2 + trailing queue-shock dose (frozen detector outputs)
H4 = H3 + Hyperliquid short-horizon own-venue prefix
```

核心比较仍然是 H1 vs H3：cross-spread 状态之外，queue-shock 结构是否
提供稳定的 hazard 预测增量。

### RQ3：regime dwell time vs 延迟预算（可执行性）

策略能利用的不是单次事件的提前量，而是"危险状态持续得比反应慢"。

定义预测 hazard 高于分位 \(q\) 的时段为 high-hazard regime，测量：

- high/low regime 的 dwell time 分布（p10/p50/p90）；
- regime 切换率；
- 与假设延迟档位 `25/50/100/250/500ms` 的对比（沿用 v1 Gate E 档位）；
- 进入/退出 regime 时刻的 hazard 路径形状（切换是否可提前观测）。

已有的先验证据：Jul30 cluster 时长 p50 ≈ 80ms、flow 时长
p50 ≈ 477ms / p90 ≈ 2.5s——flow 尺度的持续期显著长于典型撤单延迟，
说明该问题值得测量而非先验否定。

三问的 go/no-go 链：

```text
RQ1 fail -> 静态姿态结论，停止
RQ1 pass, RQ2 fail -> 时变但不可预测，停止或转向数据扩充
RQ1+RQ2 pass, RQ3 fail -> 可预测但不可执行，记录并转向更低频姿态问题
RQ1+RQ2+RQ3 pass -> 进入姿态策略设计（仍为 public-data，无订单）
```

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

## 6. Adverse event 与 outcome 合同

### 6.1 参考报价与 adverse event

对侧 \(s\)、距离 \(\delta\in\{0,\ 1\ \text{tick}\}\)（at-best 与 best±1）
定义假设参考报价位。adverse event 沿用 v1 §9.2 的 public 证据族：

```text
public_trade_reaches_quote
public_bbo_moves_through_quote
```

命名纪律沿用：字段名不得含 `fill/filled/execution_pnl/own_order`。

### 6.2 Horizon 与采样

预测 horizon：

```text
h ∈ {50, 100, 250, 500} ms
```

1000/2000ms 只作为 descriptive atlas，不进入 primary gates（依据：
2000ms 尺度独立块数 Jul30=9、Aug04=6，不支持校准声明）。

hazard 标签在 common timeline 的评估网格上构造；评估网格点必须携带
strict-as-of source timestamp 与 no-new-information flag；重复
forward-filled 状态不计为新观测。

### 6.3 Censoring

完整继承 Stage 4 已验收的 interval/right/segment/quality censoring 机制。
Hyperliquid timing 保持 interval censoring，禁止点化；horizon 端点落入
censored 区间的标签显式标记为 interval-ambiguous，不得静默取边界。

## 7. 估计器合同

按复杂度升序，前一层是后一层的 baseline：

1. **经验分箱 hazard**（cross-spread bins × dose bins × side）：
   可完全审计，是 RQ1 的 primary estimator；
2. **正则化 logistic / discrete-time hazard regression**：
   H0–H4 嵌套消融的 primary estimator；
3. **quantile / 非线性模型**：仅作 robustness，不得成为唯一支持证据。

神经网络与深度序列模型在本框架 v2 内不使用。

case-retrieval（v1 §11）在连续框架下不再是 primary estimator；若后续
恢复使用，需单独命名合同。

## 8. 评估与统计单元

- **within-session**：walk-forward（按 segment 或时间块），禁止随机
  train/test 切分；
- **不确定性**：flow-block / time-block bootstrap，禁止行级 IID bootstrap；
- **复制单元**：session。跨 session 声明的证据强度被独立采集日数封顶，
  报告必须显式给出该封顶；
- **评分**：binary hazard 用 log loss + Brier + reliability table；
  分层单调性用预测十分位的 realized rate；dwell time 用分布分位数；
- **降级发布**（继承 Stage 2 纪律）：所有结构计数标注
  `never interpreted as universal N_eff`。

evidence label 沿用 v1 §4 词表，新增：

```text
prospective_holdout   （v2 冻结后新采集的 session，见第 12 节）
```

## 9. 资产继承与降级表

| v1 / Stage 1–4 资产 | v2 中的地位 |
| --- | --- |
| common L2 timeline（R0/R1） | 主数据平面，直接复用 |
| frozen trigger contract + parity（Stage 3） | dose 特征生成器 + 验证锚点，阈值不变 |
| Stage 2 density/merging/sensitivity package | RQ1/RQ3 的直接证据输入 |
| Stage 4 Episode v3 Family A/B（1.5GB package） | landmark view + 交叉核对源；不重开 |
| observed-at feature ledger 机制 | 全量继承 |
| interval/right/quality/segment censoring 机制 | 全量继承 |
| v1 B0–B4 嵌套问题 | 移植为 H0–H4 |
| v1 Gate E 延迟档位 | 移植为 RQ3 延迟预算档位 |
| per-episode Palm 统计 | 降级为诊断视图 |
| case retrieval as primary estimator | 移出 primary，恢复需新合同 |
| v1 Ordered Queue Stage 5+ | 停止按 v1 编号推进 |

## 10. Stage H0：hazard 时变性与 dwell-time 审计（首个可派发单元）

v2 的第一个执行单元是一个**只读、小输出**的审计任务，消费已验收
packages，不新建大规模数据面。它同时是 trust kernel（见第 13 节）的
首个试点消费者。

输入（只读）：

- Jul30 / Aug03 / Aug04 accepted R0/R1 timeline；
- Stage 2 density/merging/sensitivity package；
- Stage 4 Episode v3 package（仅交叉核对）；
- Aug07 event rows 在本 stage **不打开**。

产出：

1. 各 session、各 side、各 horizon 的无条件 adverse-event rate 路径；
2. 粗状态分箱（cross-spread × dose）的经验 hazard 表与单调性；
3. 经验 hazard 十分位分层的 realized rate spread（RQ1 primary 证据）；
4. high/low hazard regime 的 dwell time 分布 vs 延迟档位（RQ3 初值）；
5. 各 horizon × session 的依赖块数与可行性表（100/250/500/1000/2000ms
   全档位 overlap block 计数，补齐 Stage 2 仅 2000ms 的缺口）；
6. H0 vs H1 的粗校准差（RQ2 的先导信号，仅分箱估计器）。

明确非目标：不拟合正式模型、不做模型选择、不触碰 Aug07、不改任何
已验收 artifact。

Stage H0 的结果直接决定 v2 主建模阶段的派发与否及其 primary horizon。

## 11. 预注册 gates 与决策出口

以下数值在 v2 contract 冻结时最终确定；本稿给出默认值。

### Gate H-A：时变性（RQ1）

- 经验 hazard 十分位分层中，top-decile 与 bottom-decile 的 realized
  adverse rate 比值在 primary horizon 上 ≥ `3`，且分层大体单调；
- 在 ≥2 个 session 上成立，且无 session 出现实质反向；
- 不确定性以 flow-block bootstrap 报告。

### Gate H-B：可预测性与增量（RQ2）

- H1 在 log loss/Brier 上稳定优于 H0；
- H3 相对 H1 与 H2 的改善为正、跨 ≥2 session 保持、
  block bootstrap 下保留、不来自单一事后状态桶
  （完整移植 v1 Gate D 措辞）；
- 评分度量与 normalization 在冻结时点名（修复 v1 Gate C 的
  "5% worse" 未定义问题）。

### Gate H-C：可执行性（RQ3）

- high-hazard regime dwell time 的 p50 在 primary horizon 上大于
  所选延迟档位；
- regime 进入时刻在延迟档位内可观测（进入前特征已可见）；
- 结果按 side/session/underlying-regime 分层报告。

### 决策出口（唯一 primary classification）

```text
continuous_hazard_maker_posture_supported
  RQ1/RQ2/RQ3 全过；可进入 public-data 姿态策略设计阶段。

hazard_flat_static_posture_indicated
  时变性不足；结论为静态姿态（更宽/更小/不做），本方向研究完成。

hazard_variation_unpredictable
  时变但状态不可预测；转向数据扩充或特征扩充，不做策略声明。

predictable_but_not_actionable
  可预测但 dwell/延迟不匹配；记录并限定于更低频姿态问题。

cross_spread_supported_dose_increment_not_supported
  H1 成立而 H3 无稳定增量；queue-shock dose 降级为纯锚点。

cross_session_unstable_needs_more_sessions
  within-session 成立但跨 session 不稳；触发第 12 节数据扩充后重估。

inconclusive_data_quality_or_coverage
  支撑、覆盖或数据质量不足。
```

任何出口都不授权订单、私有 endpoint 或 live 部署。

## 12. 数据扩充：滚动采集与 prospective holdout

连续框架解决概念错配，不解决独立日数不足。跨 session 校准声明的证据
强度由独立采集日数封顶（当前为 4）。因此：

1. v1 §20 的 "does not collect new data" non-goal 在 v2 中**显式撤销**，
   替换为一个常态化滚动采集组件；
2. 采集使用已验收的 collection + QA 管线，目标节奏示例：每周 3–4 个
   session、覆盖不同 time-of-session 与 underlying（KRX）regime，
   持续至少数周；
3. 每个新 session 在 v2 contract 冻结之后采集，天然具备
   `prospective_holdout` 资格；其准入沿用既有 collection QA、
   immutable raw audit 与 versioned postprocess 验收链；
4. `cross_session_unstable_needs_more_sessions` 出口的重估只能使用
   prospective holdout sessions，不得回收 discovery sessions。

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
- 优化 GLFT 参数或任何报价参数；
- 部署或授权 live 策略；
- 在 Stage H0 阶段打开 Aug07 event rows；
- 将结构行数解释为统计样本量。

## 15. 完成定义

v2 研究完成当且仅当 accepted packages 回答：

1. SKHYNIX 目标 venue 的短 horizon adverse hazard 是否显著时变？
2. 该时变是否可由决策时点可观测状态预测？queue-shock dose 是否在
   cross-spread 之外提供稳定增量？
3. 高危状态的持续时间是否长于现实延迟预算，使连续姿态策略在原则上
   可执行？
4. 上述结论是否在 ≥2 个 session 上稳定，并在 prospective holdout
   sessions 上保持？
5. 由此得出的策略形态结论是什么：连续姿态策略、静态姿态、或不适合
   在该标的做 maker？

在第 11 节某一出口被正式接受之前，本项目可以发布 auditable 的 hazard
审计与 descriptive atlas，但不得声称任何可部署的 maker 规则。

## 16. 相对 v1 的变更记录（changelog）

| 变更 | v1 | v2 | 依据 |
| --- | --- | --- | --- |
| 统计框架 | trigger-aligned episode / Palm | 连续 conditional hazard | Stage 2 密度证据（§2） |
| trigger 角色 | 抽样原点 + 特征 | dose 特征 + 验证锚点 | 同上 |
| primary horizon | 至 2000ms | ≤500ms；1000/2000ms 降级 descriptive | 2000ms 块数 Jul30=9 / Aug04=6 |
| primary estimator | case retrieval | 分箱 hazard + 正则化回归 | 支撑不足以校准高维近邻 |
| 增量问题 | B0–B4 | H0–H4（移植） | 保留核心科学问题 |
| actionability | per-episode margin | regime dwell vs 延迟预算 | 连续过程无单事件提前量 |
| 新数据 | non-goal | 滚动采集 + prospective holdout | 独立日数是硬约束 |
| 验收架构 | 属性列表 | trust kernel + surface matrix + preflight | Stage 4 复盘 §11/§14/§16 |
| Gate 度量 | "5% worse" 未定义 | 冻结时点名评分与 normalization | 复盘教训 |
