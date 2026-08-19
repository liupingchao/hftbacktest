# 开发框架升级路线图：从可信产物到有生命的知识

Date: 2026-08-17

Status: reference roadmap for the repo owner; revisit before dispatching each
upgrade step; no step is dispatched by this document itself

读者：总控（人）与未来所有规划 session。本文档回答一个问题：

```text
在 trust kernel（产物可信）与 workflow 三权分立（过程有纪律）之后，
这个 repo 的下一个层次是什么，按什么顺序、做哪些具体的事到达。
```

一句话答案：**让知识有生命** —— 让研究结论成为带保真度、带保质期、
带依赖链的一等资产；让保真度阶梯和证伪闭环替人决定
"什么该被相信、相信多久、什么时候必须重新怀疑"。

---

## 0. 现状诊断（2026-08-17）

已经拥有的：

| 能力 | 载体 | 状态 |
| --- | --- | --- |
| 产物可信 | trust kernel + surface matrix（复盘 §14/§16） | 已规划，未抽取 |
| 过程纪律 | workflow-kit：方案/派发/QA/复盘 | 运行中，成本已优化过一轮 |
| 数据真值 | 采集 QA 链、immutable raw、R0/R1、content addressing | 成熟 |
| 事实记录 | `findings.md` | **7,824 行 / 559 sections，见下** |
| live 基础 | T068 系列：live account control、seeded canary、preflight | 存在但与研究脱节 |

三个结构性缺口：

1. **研究结论以散文存在。** `findings.md` 是 append-only 时间序日志，
   被 `AGENTS.md` 列为必读，但 7,800 行没有 session 真正读得完；
   没有任何机制保证"制定新方案时，相关事实自动浮现"。
   Stage 2 密度证据推翻 v1 框架靠的是人恰好重读了它。
2. **真值来源之间没有形式化阶梯。** 数学不变量、历史重放、backtest、
   shadow、canary、live 的信任语义完全不同，但"某结论在哪一层被验证过"
   只存在于报告叙述里，命名纪律（如 "must never be labelled observed
   cancel effectiveness"）靠人工维持。
3. **live 与研究是两个世界。** 采集→研究→结论→live 的每一次跨越
   都靠总控人工摆渡；live 观测不自动回流证伪任何研究结论。

五个升级步骤按依赖排序，每步消费前一步：

```text
Step 1  Trust Kernel + Surface Matrix 模板     （信任的地基）
Step 2  Claim Registry v0                      （知识变资产，最高杠杆）
Step 3  Parity Harness：one kernel, many envs  （研究与 live 同一对象）
Step 4  Fidelity Gates 进 workflow 模板        （跨层晋升成为制度）
Step 5  监控即证伪                              （闭环，知识自动过期）
```

---

## Step 1：Trust Kernel + Surface Matrix 模板

**解决的问题**：Stage 4 六轮 QA 证明，publication/admission 信任边界
若不抽成共享组件，每个 stage 都要在 GB 级数据上重新讨价还价；
教训若不进模板，下一个 session 必然遗忘。

**交付物**：

1. 共享模块（建议 `tools/research_trust_kernel/`，独立于任何 stage）：
   - exact `lstat` tree universe（file/dir 白名单，symlink/FIFO/socket/
     device 一律 fail closed，root 类型先于路径解析检查）；
   - source-owned aggregate evidence contract 校验
     （exact projection universe、exact key universe、canonical
     integer/SHA policy、zero-mismatch policy）；
   - durable evidence gate：任何 `unchanged` 声明必须先指向
     repo 内可重放的 prior inventory，`/tmp` 证据一律拒绝；
   - 分层 package identity：
     `research_data_identity / runtime_contract_identity /
     publication_envelope_identity`，trust-only repair 不重建数据层；
   - fail-closed admission CLI，供业务与 QA 双方调用同一入口。
2. kernel 自己的 negative test 矩阵：把第六轮 QA 的
   `98` aggregate 负测试、`36` tree attacks、`12` production attacks
   移植为 kernel 永久测试，**全部在秒级 fixture 上运行**。
3. parity 证据：用 kernel 对已验收 Stage 4 package 重放完整 admission，
   要求与第六轮 QA 判定逐项一致（这是"抽取未改变信任边界"的机器证明，
   与 Stage 3 detector parity 同一纪律）。
4. 模板改造（`.workflow/workflow-kit/task-dispatch-template.md`）：
   - 新增必填节 `Load-Bearing Surface Matrix`：
     每个 output family 一行：
     `surface / authoritative source / decision time / exact fields /
     rebuild oracle / negative mutation / durable evidence`；
   - 新增必填节 `Trust Kernel Version`：pin accepted kernel 版本；
   - QA 模板新增第一条检查：矩阵是否完整定义 ——
     **任一 output family 无对应 negative mutation 即直接拒绝，
     即使全部 positive tests 通过**（复盘 §16 第 7 条的机械化）。
5. hostile preflight 规则进模板：全部 negative 场景先在秒级 fixture
   上失败关闭，才允许第一次全量构建。

**验收标准**：kernel 通过一次独立 QA；对 Stage 4 包的 parity 逐项一致；
之后任何 stage 的 QA 不再重新审理 kernel 覆盖的信任面。

**卫生任务顺带**：清理 `local_live_analysis/` 下 7 个空的
`...episode_v3 2`–`8` 残留目录；为 1.5GB Stage 4 包定 durable 归档位置
（amdserver），worktree 不是它的家。

---

## Step 2：Claim Registry v0 —— findings.md 的结构化继任者

**解决的问题**：知识不流动。7,824 行的 findings.md 有三个病：
读不完（必读文件形同虚设）、检索不能（新方案不知道该考虑哪些事实）、
不会过期（三周前的市场事实和永久的工程教训混在一条时间线上）。

**核心设计：把"事实"拆成两类，分开治理。**

`findings.md` 里混着两种完全不同性质的条目：

| 类型 | 例子 | 性质 | 归宿 |
| --- | --- | --- | --- |
| 工程教训 | "`is_file()`/`is_dir()` 会漏 dangling symlink" | 永不过期，应变成代码 | **negative test / kernel / 模板**（Step 1 已给了它们家） |
| 经验命题 | "Jul30 触发中位间隔 29ms"、"收紧触发不改变 block 数" | 有 scope、有保质期、可被证伪 | **claim registry** |

工程教训的正确终点不是文档而是机器强制（棘轮原则）；
经验命题的正确载体是下面的 claim schema。

**交付物 1：claim schema 与目录**

```text
claims/
  registry.jsonl          # 一行一 claim，append-only，机器可读
  claims.md               # 由 registry 生成的人类可读视图（勿手编）
  schema/claim_schema.json
```

每条 claim：

```json
{
  "claim_id": "CLM-0042",
  "version": 1,
  "statement": "SKHYNIX Jul30 Family-A inter-trigger p50 = 29.1ms; trigger process is near-continuous",
  "statement_zh": "Jul30 触发中位间隔 29ms，触发过程近连续",
  "kind": "market_fact | method_result | infra_fact | negative_result",
  "scope": {
    "instrument": "SKHYNIX",
    "sessions": ["jul30"],
    "horizon_ms": null,
    "topology": "local_collector_v?"
  },
  "fidelity_level": "L1",
  "evidence": [
    {"package": "stage02_density", "sha256": "...", "field": "inter_trigger_distribution.csv"}
  ],
  "depends_on": [],
  "supersedes": null,
  "expiry_policy": "refresh_on_new_session | never | date:2026-09-15",
  "status": "active | expired | falsified | superseded",
  "tags": ["trigger_density", "episode_framework", "hazard"],
  "created_by_task": "0815T001",
  "notes": "推翻 v1 稀有事件假设的核心证据"
}
```

设计要点：

- `statement` 必须是**可证伪的命题**，不是叙述。写不成命题的内容
  不进 registry（它属于报告或教训）。
- `kind: negative_result` 是一等公民——"什么不成立"（如
  "收紧阈值不制造独立性"）往往比正结果更能阻止未来浪费。
- `evidence` 指向 accepted package 的哈希，接 Step 1 的信任链：
  **claim 的可信度 = 其 evidence 的 admission 状态**。
- `expiry_policy` 表达非平稳性：市场事实默认
  `refresh_on_new_session`，数学/机制事实 `never`。

**交付物 2：findings.md 的迁移与瘦身**

一次性迁移任务（可以派发给业务线程，QA 验收覆盖率）：

1. 逐 section 分类 559 个条目：工程教训 → 开 issue 转化为
   negative test/模板条款后归档；经验命题 → 提炼为 claims；
   纯过程叙述 → 保留原文，仅归档。
2. `findings.md` 冻结为 `docs/findings_archive_2026H1.md`，只读。
3. 新的 `findings.md` 只保留一个瘦身角色：**尚未结晶为 claim 或
   negative test 的临时观察缓冲区**，并规定每个任务收尾时必须清空
   （结晶或丢弃）——它从"越写越长的终点"变成"必须清空的中转站"。

**交付物 3：智能提示——方案制定时自动浮现相关 claim**

这是你点名的痛点，机制分三层，从便宜到贵：

1. **强制声明（纯模板，零代码）**：派发模板新增必填节：

   ```text
   ## Claim Dependencies
   - consumes:   本任务假设成立的 claims（列 ID）
   - may_falsify: 本任务可能证伪的 claims
   - may_produce: 预期产出的新 claims（kind 与 scope 草案）
   ```

   QA 验收对应新增检查：产出结论是否已写回 registry；
   consumes 里的 claim 是否全部 `active`（引用 expired/falsified
   claim 直接拒绝）。**仅此一层就已解决"制定方案时不考虑既有事实"
   的 80%**，因为它把回忆义务从"读 7800 行"变成"检索一个结构化表"。
2. **检索工具（一个小 CLI）**：

   ```text
   tools/claims/query.py --tags trigger_density --scope-instrument SKHYNIX \
       --status active --format md
   ```

   总控起草方案时先跑一次相关 tag 的查询，把命中表贴进派发文档。
   registry 是 jsonl，实现是几十行；关键是让"查一下"比"想不起来"
   更便宜。
3. **AI 辅助预检（可选，最后做）**：起草方案后让一个只读 session
   拿方案全文对 registry 做相关性匹配，输出
   "这个方案可能忽略的 active claims / 与哪个 claim 冲突"。
   这一步价值真实但不必先做——前两层是结构，这层只是润滑。

**立即可回填的种子 claims**（迁移任务的第一批，全部已有 accepted 证据）：

- Jul30/Aug03/Aug04 触发密度与 inter-trigger 分布（stage02）；
- "named sensitivity 收紧不改变 2000ms overlap block 数"（stage02，
  negative_result）；
- 2000ms 独立块数 Jul30=9 / Aug03=232 / Aug04=6（stage02）；
- flow 时长 p50≈477ms / p90≈2.5s（stage02，RQ3 先验证据）；
- Stage 4 package 身份与 cardinality 事实（stage04）；
- v2 框架的三条设计公理（框架文档 §2，标 `depends_on` 上面的密度 claims
  ——这会立刻示范依赖链的价值：密度 claim 若被新 session 证伪，
  v2 公理自动标疑）。

---

## Step 3：Parity Harness —— one kernel, many environments

**解决的问题**：backtest 里验证的与 live 里跑的是不是同一个东西，
目前靠人担保。v1 文档的警告——"不要另外写一个研究版 detector，
那个双胞胎最后大概率会翻脸"——要从警句升级为架构约束。

**架构约束**：

```text
决策核（纯函数，无 IO，无时钟）：
  decode → book build → feature → detector/hazard → policy decision

环境适配器（唯一被允许持有 IO/时钟的层）：
  replay_env    喂历史 raw journal，虚拟时钟
  backtest_env  喂仿真撮合（hftbacktest 本体）
  shadow_env    喂 live feed，决策只记录不发送
  live_env      喂 live feed，决策发送
```

**交付物**：

1. 决策核抽取为纯模块（Stage 3 detector 抽取已开先例，照同一模式：
   行为不变提取 + 全量历史 parity 证明）；
2. **录制-重放 parity gate**：shadow/live 会话必须录制完整原始输入流
   （raw payload + 本地单调时戳 + connection epoch，M1 journal 契约
   已定义过）；离线用 replay_env 重放，要求**逐决策 byte-identical**；
3. parity gate 成为常驻 CI 级检查：每次决策核代码变更，对一个
   固定的 golden journal 重放，diff 非空即 fail；
4. 每个 shadow/live 会话结束后自动跑一次当日 parity，结果写入
   session 的 QA 证据。

**与 Step 2 的接口**：parity 通过是 claim 从 L1/L2 晋升 L3 的
准入条件（见 Step 4）；parity 失败自动降级所有依赖该决策核版本的
L3+ claims。

**顺序理由**：放在 Step 2 之后、任何 v2 shadow 工作之前——
v2 Stage H0 是纯离线审计不需要它，但 H0 之后若 RQ1–RQ3 通过，
下一步必然是 shadow，届时 parity harness 必须已就位。

---

## Step 4：Fidelity Gates 进 workflow 模板

**解决的问题**：真值来源的信任语义差异目前靠命名纪律人工维持。
把它变成 claim 的类型系统与晋升制度。

**保真度阶梯定义**（写入 workflow-kit 手册）：

```text
L0  数学不变量/市场机制    证明逻辑一致性；永不证明经验命题
L1  历史重放（确定性）      证明"当时的 view 下会看到什么"；无反事实
L2  backtest 仿真          证明模型化 fill 假设下的反事实；不证明 fill 模型
L3  shadow live            证明生产拓扑下的实时可观测性与延迟真值；无 own fill
L4  randomized canary      证明小规模因果 EV；不证明规模化
L5  live champion          证明一切；样本最贵
```

**晋升规则（每条都是 QA 可检查的 gate）**：

| 晋升 | 前置条件 |
| --- | --- |
| →L1 | evidence 指向 accepted package（Step 1 admission 通过） |
| L1→L2 | fill/latency 模型假设显式列出并各自成为带 expiry 的 claim |
| L2→L3 | Step 3 parity gate 通过 + 生产拓扑记录在 scope 里 |
| L3→L4 | **总控人工审批（涉及真钱，永不自动）** + deterministic 随机化方案 + 预注册评估窗口 |
| L4→L5 | canary 期满 + 预注册 EV gate 通过 + 总控审批 |

**降级规则（自动，不需要人）**：

- 上游 claim `falsified/expired` → 下游依赖 claim 自动标 `suspect`；
- parity 失败 → 该决策核版本的所有 L3+ claims 自动 `suspect`；
- scope 漂移（换标的、换拓扑、换 fee tier）→ claim 不迁移，
  新 scope 需要新 claim。

**模板改造**：

- 派发模板 `may_produce` 节必须声明目标 fidelity level；
- QA 模板检查：结论措辞与 level 匹配（L1 结论不得用 L3 词汇——
  这把 v1 "must never be labelled observed cancel effectiveness"
  的命名纪律从人工审查变成机械检查）；
- 任何 live 参数变更的派发文档必须列出其依赖的 active claims 及
  各自的 fidelity level——**live 配置从此可追溯到证据链**。

---

## Step 5：监控即证伪 —— 闭环

**解决的问题**：live 是最高保真的采集器，但目前它的观测不回流。
claim 的 expiry 只按日历，应升级为按证据。

**交付物**：

1. **校准监控**：对每个进入 L3+ 的 claim 定义一个 live 可计算的
   监控量（如：预测 hazard 十分位 vs realized rate；假设延迟档位 vs
   实测 cancel 延迟分布；basis 残差分布 vs 历史）。监控定义与
   容忍带在 claim 创建时预注册，不得事后调整。
2. **自动降级**：监控量出带 → claim 自动 `suspect` → 依赖链传播 →
   生成一条研究任务草案（"CLM-00xx 在 live 上漂移，需要新 session
   数据重估"）进入总控的待办队列。
3. **canary 即实验**：v1 设计的 hash-based deterministic 随机化
   在此复活——canary 的 control/treatment 结果自动写回 registry
   成为 L4 evidence，而不是躺在报告里等人读。
4. **滚动采集接入**（v2 §12 的滚动采集组件）：每个新 prospective
   session 验收后，自动跑一遍所有 `refresh_on_new_session` 的
   claims 的重估脚本，结果决定 refresh 或 falsify。

到这一步，系统的日常运转变成：

```text
知识库自己暴露缺口（expired/suspect claims）
  → 缺口生成任务草案
  → 任务在阶梯上产出带标签的新 claim
  → live 持续考验所有 active claims
```

总控收缩到三件机器不能做的事：**选研究方向、审批跨层晋升
（尤其 L3→L4 的真钱门槛）、修宪（改框架本身）**。

---

## 6. 执行顺序与依赖总览

```text
Step 1 kernel+matrix ──┬─→ Step 2 registry ──→ Step 4 fidelity gates ──→ Step 5 闭环
                       └─→ Step 3 parity harness ──↗
```

| Step | 规模 | 何时启动 | 阻塞谁 |
| --- | --- | --- | --- |
| 1 kernel + 模板 | 中（抽取+parity+模板） | 立即；先于任何 v2 数据面 | 一切 |
| 2 registry v0 + findings 迁移 | 小（schema+CLI）+ 中（559 条迁移） | Step 1 派发后即可并行起草 schema | Step 4/5 |
| 3 parity harness | 中 | v2 Stage H0 之后、shadow 之前 | L3 晋升 |
| 4 fidelity gates | 小（纯模板+手册） | Step 2 落地后 | Step 5 |
| 5 监控即证伪 | 中 | 首个 L3 claim 出现后才有意义 | — |

与 v2 研究的交织：Step 1 → v2 Stage H0（作 kernel 首个试点消费者）
→ Step 2 迁移（H0 产出直接以 claim 形式落地，作 registry 首个
生产者）→ H0 gates 通过则 Step 3 → shadow。

**反模式警告**（每步都要抵抗的诱惑）：

- 不要给 registry 加 web UI / 数据库——jsonl + CLI 足够，
  形式越重越没人维护；
- 不要把所有旧 findings 都硬塞成 claim——写不成可证伪命题的
  就让它留在 archive 里；
- 不要让 fidelity gates 拖慢 L0/L1 的探索——阶梯的刚性应随
  层级递增，L1 研究保持现在的速度；
- 不要在没有 L3 claim 之前做 Step 5——监控没有对象时是纯开销。

## 7. 每步的"完成"定义

| Step | 完成标志 |
| --- | --- |
| 1 | 某个新 stage 的 QA 报告里出现"kernel vX admission 通过，信任面不再重审" |
| 2 | 某个新派发文档因引用了 expired claim 被 QA 拒绝（制度第一次咬人） |
| 3 | golden journal 重放进 CI；某次决策核改动被 parity diff 拦下 |
| 4 | 某条 L1 claim 因措辞越级（用了 L3 词汇）被 QA 打回 |
| 5 | 某条 claim 因 live 漂移自动变 suspect 并生成研究任务草案 |

注意这些完成标志全部是**制度第一次实际发挥作用的时刻**，
而不是"代码写完的时刻"——框架的存在性由它拦下的第一个错误证明。
