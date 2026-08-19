# SKHYNIX Stage 4 Episode v3 执行与返修复盘

日期：2026-08-17

复盘范围：

- workflow task：`0815T003`
- Ordered Research Queue 第 4 项：
  `Build Family A and Family B episode v3 for Jul30 only`
- 执行时间：2026-08-15 至 2026-08-16
- 复盘原因：经历 5 次有界返修、6 轮独立 QA 才完成验收
- 当前状态：Stage 4 已通过；Stage 5 前置门槛已满足，但按用户要求暂停派发

## 1. 总结论

第六轮独立 QA 已于 2026-08-16 给出最终结论：

- 状态：`已通过`
- P0/P1/P2/P3：`0/0/0/0`
- QA report/mirror SHA256：
  `0b85ae527f7e91f40f3e4401969596ff6462964a7e05cb8d0de73e763ad3e592`

最终接受的 package 身份为：

- files/artifacts/bytes：
  `107 / 106 / 1,561,307,420`
- core：
  `78be6559c7ac5aaec4d5411b042f7ddcce522473f60f987237bcaa9d2dd5e157`
- full inventory：
  `669fb7d12f25cfa7828aec0fb1546398b1def754952de2290cd19784a477a433`
- frozen contract：
  `b80b9bae2d6cf18cb6e7be4f133467138527fec20eef7054ab38b7f8c70cefde`
- manifest：
  `2c802336c1446eaf50eaf5ef546b115046abe3a1b69fdd1fcaea31d22a8e64a6`

这次通过不能解释为“原方案从一开始就是完整的”。更准确的结论是：

```text
研究对象与总体方向基本正确。

Stage 4 的实施与验收架构不完整，
独立 QA 通过五轮连续攻击，才把缺失的证明契约补齐。
```

原方案对以下研究问题设计得较好：

- Family A 保留全部 Candidate，避免 confirmation selection bias；
- Family B 只作为同一 Candidate 的 Confirmed-time linked view；
- Candidate 与 Confirmed 两个 landmark 不混淆；
- 每个 feature 有 observed-at 与 source provenance；
- Hyperliquid public timing 使用 interval/right censoring；
- 不越界到订单、成交、PnL、模型和 actionability。

但方案没有在 Stage 4 开工前冻结一个统一、机器可执行的
adversarial acceptance model。结果是下面五层 trust boundary 被逐轮补齐：

1. 研究语义与 detector 语义；
2. 完整 source-derived output surface；
3. standalone evidence attestation；
4. 跨返修 durable evidence；
5. 完整 filesystem/package namespace identity。

第一轮返修修正了真实 Episode 数据语义。后续大部分返修没有继续改变
research rows，而是在补强“这个 package 如何证明自己确实等于 source
truth”。

## 2. Stage 4 原始目标

Stage 4 要建立一个共享 Candidate record，并发布两个 linked views：

- Family A：
  Candidate-aligned，覆盖全部 `268,522` candidates，其中包括
  `126,754` rejected/unconfirmed cases；
- Family B：
  Confirmed-time view，只覆盖同一 Candidate IDs 中的
  `141,768` primary confirmed cases。

计划要求 package 包含：

- shared anchors；
- exact sparse path；
- Candidate/Confirmed fixed grid；
- event-count views；
- per-feature observation ledger；
- interval/right/quality/segment-censored public-market outcomes；
- immutable source lineage；
- deterministic package identity；
- public-market-only boundary。

研究对象始终是：

```text
P(R, O_market | S_pre, T)
```

本阶段不授权：

```text
P(O_own(a) | S_pre, T)
KEEP/CANCEL_RISK_SIDE causal EV
真实 fill probability
fee / inventory / PnL
live execution
```

## 3. 最终接受的数据事实

最终 package 的核心 cardinality：

- anchors / Family A / outcomes：
  `268,522 / 268,522 / 268,522`
- Family B / confirmed：
  `141,768 / 141,768`
- rejected Family A：
  `126,754`
- feature rows A/B：
  `23,092,892 / 15,310,944`
- views A/B：
  `268,522 / 141,768`
- fixed-grid rows A/B：
  `4,564,874 / 2,410,056`
- event-count rows A/B：
  `805,566 / 425,304`
- sparse-range rows A/B：
  `268,522 / 141,768`
- clusters / flows / 2,000ms overlap blocks：
  `39,928 / 10,536 / 9`

Outcome/censoring：

- interval-censored：
  `627,406`
- right-censored：
  `177,826`
- segment-censored：
  `316`
- quality-censored：
  `18`
- epoch-censored：
  `0`
- point-coerced interval：
  `0`

最终 invariant：

- future feature observation mismatch：
  `0`
- anchor ordering mismatch：
  `0`
- cross-segment / cross-epoch path mismatch：
  `0 / 0`
- synthetic rejected confirm：
  `0`

Durable research-data inventory：

- research files：
  `99`
- research bytes：
  `1,560,514,934`
- canonical inventory SHA256：
  `bb5aed2099b1a97da5b476d4b09dfe4a7bac06f0bf331f8ca864a88daf5c9232`
- evidence：
  `.workflow/reports/0815T003-round4-pre-repair-research-inventory.csv`
- evidence file SHA256：
  `07521dbc2c8bfe5f41a3ab14ac5493591dad878d52053b2d74aad251b9f022df`

## 4. 完整执行时间线

| 阶段 | 业务报告 | 独立 QA | 结果 | 新发现的主要缺陷 |
| --- | --- | --- | --- | --- |
| 初始构建 | `0815T003-business.md` | QA Round 1 | 失败：`P1=4, P2=1` | detector provenance、burst、interval、source semantics、strict-pre |
| 返修 1 | `0815T003-business-r1.md` | QA Round 2 | 失败：`P1=1` | exact replay 没覆盖完整 features/views |
| 返修 2 | `0815T003-business-r2.md` | QA Round 3 | 失败：`P1=1` | standalone 接受残缺且矛盾的 aggregate evidence |
| 返修 3 | `0815T003-business-r3.md` | QA Round 4 | 失败：`P1=1, P2=1` | aggregate extra key；缺少 durable prior inventory |
| 返修 4 | `0815T003-business-r4.md` | QA Round 5 | 失败：`P1=1` | dangling symlink 不在 artifact/full inventory universe |
| 返修 5 | `0815T003-business-r5.md` | QA Round 6 | 通过：全零 | exact `lstat` tree closure 被独立接受 |

五轮失败 QA 共记录：

- P1 findings：
  `8`
- P2 findings：
  `2`
- distinct findings：
  `10`

Stage 4 focused tests 的增长：

- 初始：
  `12`
- 返修 1：
  `17`
- 返修 2：
  `46`
- 返修 3：
  `80`
- 返修 4：
  `81`
- 返修 5 / 最终：
  `97`

最终验收证据：

- current related tests：
  `396 passed`
- archived Stage 4：
  `97 passed`
- aggregate negative calls：
  `98/98` fail closed
- direct filesystem attacks：
  `36/36` fail closed
- production package filesystem attacks：
  `12/12` fail closed

## 5. 初始构建复盘

初始 package：

- core：
  `14d945253ecae82cdb73bdc365ae01ae6f614b8ed7fc9ea56e08d65c4ba77398`
- full：
  `25c02766e4eb263d4ba4bf20cfe48b37a9812f20d7eb402ddbcffb7ffe450785`
- contract：
  `2239a170e670b8af7bca901524101b0beb3abc834deadfff714137f1b83d7347`
- manifest：
  `c94150a71520db7db1f1bba36f9de5716514d1e3cff4cf12299bebe271d08266`
- files/bytes：
  `107 / 1,562,104,138`

初始实现已经正确完成的部分：

- Family A 保留全部 candidates/rejected；
- Family B 与同一 Candidate identity linked；
- 核心 artifact cardinality 正确；
- Formal/Build A/Build B deterministic；
- 没有读取 later-session future rows；
- 没有 own-order、fill、PnL、model 或 actionability 越界；
- 没有 missing/duplicate Candidate IDs；
- public first-event timing 没有被整体点化；
- accepted dependencies 与 atomic publication 已存在。

初始设计仍然失败的原因：

- 把 shape、count、hash 自洽当成了比实际更强的证明；
- `observed_at <= landmark` 只能证明“不晚于决策”，不能证明
  source event 真正生成该 feature value；
- Stage 4 重新实现了 detector 的部分语义，而不是直接消费已冻结的
  confirmation state 和 burst membership；
- first-event 通用函数没有对 post-Candidate outcome 做左截断；
- source-semantic admission 只覆盖部分示例字段，没有覆盖完整
  load-bearing research surface。

## 6. QA Round 1 与返修 1

### 6.1 Confirmation provenance

全部 `141,768` confirmed candidates 的 confirmation-derived values
绑定到了更早的 Binance BBO-as-of event，而不是 detector 首次满足确认
条件的 common-L2 state。

问题不是数值必然错误，而是 provenance 声明错误：

```text
一个值可以碰巧正确，
但它不能声称来自一个没有生成该值的 source event。
```

返修：

- confirmation feature `observed_at_ns=t_confirm_ns`；
- source event 改为 exact common timeline confirmation state；
- source book version 改为该 state 的 `common_seq`。

### 6.2 Frozen burst membership

初始实现从 burst start 一直扫描到 Confirm，重新收集所有同向 trades。
原 detector burst 在 fixed-origin gap、opposite side 或 reset 后已经结束，
后续同向 flow 被错误并入。

QA 示例：

- detector truth：
  `57`
- published：
  `73`

返修：

- 从 accepted Stage 3 burst start/end/count/qty 重建 frozen burst；
- through-confirm 只使用 frozen burst 内的 prefix。

### 6.3 First-event Candidate 左截断

`206,784` 个 interval-censored target-trade outcomes 中：

- lower `< Candidate`：
  `115,637`

这让一个 post-trigger outcome 出现了 pre-trigger support。

返修：

- event scan 只使用 `event_ts > t_candidate_ns`；
- 在首次 post-Candidate non-event 前，lower bound 固定为 Candidate；
- 保持 interval observation，不点化、不填零。

### 6.4 Source-semantic admission fail-open

QA coherently 修改并重算 package hash 后，下面的变化仍被接受：

- classification relabel；
- degraded evidence deletion；
- sparse count；
- event identity；
- markout；
- source age。

返修：

- 从 accepted Stage 2/3 + Jul30 structured source exact 重建；
- exact 比较 anchors、paths、outcomes 和 confirmation features；
- source/archive verifier 都执行 source-semantic replay。

### 6.5 Strict-pre baseline

Candidate 同 receipt timestamp 的 event 不能成为 pre-trigger baseline。

返修：

- baseline 只允许 `< t_candidate_ns`；
- equal timestamp 不进入 baseline，也不伪装成 post-trigger first event。

返修 1 package：

- core：
  `5134f1fd31333f0c4c6a972e524b03e2a34bb4435abe6beddda4c62e34e63586`
- full：
  `c581b65420a08ee2ec658369ecbfde240228127d1a70a243829fa1279feb77e6`
- files/bytes：
  `107 / 1,561,201,692`

这是唯一一轮明确同时修正多项已发布研究数据语义的返修。后续返修主要
集中在 publication/admission trust。

## 7. QA Round 2 与返修 2

返修 1 保护了第一轮 QA 点名的字段，但没有保护完整 consumer surface。

QA coherently 修改：

```text
Family A pre_binance_bid_px: 948.13 -> 948.14
```

同步更新 artifact/core/manifest 后，source/archive 仍返回：

```text
verified=true
source_semantic_verified=true
```

根因：

- exact source replay 只覆盖 `14` 个 Family B confirmation features；
- 其他 Family A/B features 只做 schema、availability 和 as-of 检查；
- complete Family A/B view payload 不在 exact projection 中。

返修 2 把 exact whole-row projection 扩展到：

- Family A features：
  `23,092,892`
- Family B features：
  `15,310,944`
- Family A views：
  `268,522`
- Family B views：
  `141,768`

每个 feature row 的完整 load-bearing fields 都进入 exact comparison：

- value；
- observed_at；
- source_event_id；
- source_book_version；
- calculation_version；
- availability_reason。

返修 2 package：

- core：
  `4fc6ce074200a3c08f7a504f0ce2412f5dad8d7ad51aa44e1799f00f6dd886c9`
- full：
  `1f3ddd61fa0bbbf3268ae6f09b303d4876e83f456d9e6ce63956b1306707dc72`
- contract：
  `7e07af13742edadddcda32784b64e398586b7cff94c7fa9fdeb2d880b41bfb5c`
- files/bytes：
  `107 / 1,561,252,064`

这一轮的核心教训：

```text
保护已发现的字段集合，
不等于闭合整个已发布 consumer surface。
```

## 8. QA Round 3 与返修 3

heavy verifier 已经可以生成完整 source-semantic evidence，但 standalone
CLI 对返回 evidence 的二次断言过弱。

QA 构造只包含一个 projection 的 aggregate，并设置：

- count contradiction；
- digest contradiction；
- `mismatch_rows=999`；
- incomplete fields。

standalone 仍然过度声明：

```text
source_semantic_verified=true
```

这不是 data-plane defect，而是 attestation-layer defect。

返修 3 建立一个 source-owned aggregate evidence contract：

- version：
  `skhynix_jul30_source_semantic_aggregate_evidence_v1`
- exact projections：
  `12`
- exact per-entry keys；
- ordered fields；
- expected row counts；
- canonical integer policy；
- canonical lowercase SHA policy；
- zero mismatch policy；
- manifest count bindings；
- `32` 类 negative tests。

返修 3 package：

- core：
  `95dac8f94b6d8223d51b91c95cdea74963e62b9a940526f740fd104b062b9f7b`
- full：
  `49cab05e39a74e6fd120088d19fcbf657739e64f8dadba33b267edb4c8d9272c`
- contract：
  `ff2b80e627415b8fc88d05a1f08c7d740c612cfdf13a26915293bc15e862b867`
- files/bytes：
  `107 / 1,561,285,739`

业务报告说明 `99` 个 research CSV/GZ 相对上一轮 byte-identical，但旧
inventory 的逐文件细节仍保留在 ephemeral `/tmp`，没有 durable archive。

## 9. QA Round 4 与返修 4

QA Round 4 新发现两项问题。

### 9.1 Aggregate extra key

`aggregate_output_counts` 会检查 required keys，但不会拒绝：

```text
unexpected=0
```

区别是：

```text
所有 required keys 都存在
```

不等于：

```text
observed key universe 与 frozen key universe 完全相等
```

### 9.2 Cross-repair evidence 不 durable

业务报告声称 research files 未变化，但下一轮独立 QA 无法从 repository
重放上一轮 `/tmp` inventory。

区别是：

```text
业务线程说 unchanged
```

不等于：

```text
独立 QA 可以根据 durable prior inventory 逐文件证明 unchanged
```

返修前，总控先持久化：

```text
.workflow/reports/0815T003-round4-pre-repair-research-inventory.csv
```

返修 4 将 aggregate contract 升级为：

```text
skhynix_jul30_source_semantic_aggregate_evidence_v2
```

冻结：

- `12` projections；
- `32` total manifest count bindings；
- `28` exact aggregate-output keys；
- 每个 key 的 expected value；
- expected mapping 只从 projection bindings 派生；
- 不维护第二套手工 allowlist。

返修 4 package：

- core：
  `77ee81b62458a0885521e42eea7db66042efa2e41e742cb943f24f2f11bb7cac`
- full：
  `fda845f542668ffb593964dfdd2e199f37761fc4cae991d45b3a3dc2bcba983a`
- contract：
  `2bc440f4db882c28ec95fccbb8f032da4d3a73214cad8763e25e57685a419d79`
- files/bytes：
  `107 / 1,561,299,482`

全部 `99` 个 research files 与 durable inventory exact。

## 10. QA Round 5 与返修 5

QA 在 package clone 中只增加：

```text
unexpected-link -> missing target
```

旧实现分别用 `Path.is_file()` 和 `Path.is_dir()` 建 universe：

- dangling symlink 不是 file；
- 也不是 directory；
- 因而不进入 artifact closure；
- 也不进入 full inventory。

真实 admission 仍返回：

```text
rc=0
verified=true
file_count=107
```

虽然 package 已存在额外的第 `108` 个 filesystem entry。

这不是研究 row defect，但它破坏 durable package trust。

返修 5 建立单一 exact `lstat` tree universe：

- package root 必须是真实 directory；
- descendant 只允许真实 regular file 或 directory；
- symlink、FIFO、socket、character/block device、other special 一律拒绝；
- type classification 不跟随 symlink target；
- directory inventory、artifact records、artifact closure、fsync 共用同一
  exact universe。

永久 tests 覆盖：

- dangling symlink；
- symlink -> existing file；
- symlink -> directory；
- FIFO；
- root symlink；
- valid nested tree。

第六轮独立 QA 额外覆盖 Unix socket。

最终 package：

- core：
  `78be6559c7ac5aaec4d5411b042f7ddcce522473f60f987237bcaa9d2dd5e157`
- full：
  `669fb7d12f25cfa7828aec0fb1546398b1def754952de2290cd19784a477a433`
- contract：
  `b80b9bae2d6cf18cb6e7be4f133467138527fec20eef7054ab38b7f8c70cefde`
- files/bytes：
  `107 / 1,561,307,420`

第六轮 QA 独立结果：

- direct current/archive tree attacks：
  `36/36`
- production current/archive tree attacks：
  `12/12`
- aggregate negative calls：
  `98/98`
- P0/P1/P2/P3：
  `0/0/0/0`

## 11. 根因分析

### 11.1 方案冻结了属性，没有冻结完整 proof obligation

原方案列出了很多正确要求：

- causal timestamp；
- source event identity；
- exact path；
- interval censoring；
- package identity；
- hard boundary。

但没有在开工前形成一张机器可执行的表，回答：

```text
哪些 published values/filesystem entries 是 load-bearing？
每一项的 authoritative source 是什么？
如何 exact reconstruct？
哪种 coherent mutation 必须失败？
如何 durable 证明返修没有改变无关数据？
```

没有这张表，每轮返修都容易只闭合当前已知缺口。

### 11.2 Producer correctness 与 publication trust 混在一个 Stage

Stage 4 实际同时承担：

1. 构建 Episode rows；
2. 验证 causal semantics；
3. 证明每个字段等于 immutable source truth；
4. 验证 standalone attestation；
5. 验证完整 package namespace；
6. 验证 deterministic/atomic publication。

这些层的 threat model 不同，不应在一个笼统的“build task”里临时展开。

### 11.3 Stage 4 与 Ordered Queue Stage 5 边界重叠

Queue item 4：

```text
Build Family A and Family B episode v3 for Jul30 only.
```

Queue item 5：

```text
Validate anchors, all-candidate outcome coverage, per-feature observation
times, interval censoring, exact paths, dependence blocks and deterministic
rebuild.
```

实际 Stage 4 QA 已验收了 item 5 的大部分内容，仅缺一个独立发布的
dependence-block validation conclusion 和 consolidated Gate A/B report。

结果：

- Stage 4 比原编号描述大得多；
- 后续 Stage 5 有重复执行风险。

### 11.4 返修完成标准是“关闭已知 finding”，不是“全局 exit checklist”

每个业务线程都正确关闭了上一轮 QA finding，并报告没有已知
P0-P3 defect。

下一轮 QA 又从相邻 trust surface 找到新问题。

业务 handoff 前缺少一个全局 checklist：

- 所有 artifact families；
- 所有 feature/view fields；
- 所有 evidence keys/types/values；
- 所有 manifest keys；
- 所有 filesystem entry types；
- 所有跨返修 durable evidence。

### 11.5 过度重视正向 determinism

Formal/Build A/Build B 从最初就可以 byte-identical。

它只能证明：

```text
实现可重复
```

不能证明：

```text
实现语义正确
verifier 覆盖完整 output surface
attestation 不可伪造
filesystem namespace 完整
```

Determinism 是必要条件，不是独立 semantic oracle。

### 11.6 Package 没有分层，导致 trust-only repair 也要重建 1.56GB

runtime source、tests、contract、report 都进入 package identity。即使只改
attestation 或 filesystem validation，也要重新构建整个 package。

报告可确认至少：

- business full builds：
  初始 + 5 次返修，每次 Formal/A/B，共 `18` 次；
- independent QA fresh builds：
  `6` 次；
- complete builds：
  至少 `24` 次。

按 accepted-era 单次约 `22` 分钟估算，仅 full build 就约等于
`8.8` 小时串行 wall time，尚未包括：

- source/archive positive admission；
- production coherent-rehash attacks；
- full data audits；
- test suites。

最终结果可信，但达到结果的成本明显偏高。

## 12. 原方案做对的地方

### 12.1 Family A 正确解决 confirmation-selection bias

所有返修都没有通过删除 rejected cases 来简化问题：

- Family A：
  `268,522`
- rejected/unconfirmed：
  `126,754`
- Family B：
  `141,768`

### 12.2 Trigger 被正确定位为时间配准原点

Candidate 与 Confirmed 始终是同一 record 的两个 landmarks，而不是两个
独立数据集。

因此可以研究：

```text
P(O | candidate)
```

与：

```text
P(O | candidate, confirmed=1)
```

### 12.3 Public-feed timing 没有被制造成虚假精度

Hyperliquid timing 始终保留 interval/right censoring，没有为了模型方便
强制转成 point timestamp。

### 12.4 Hard boundaries 始终守住

全部轮次均未发生：

- Aug03/Aug04 future rows 读取；
- Aug07 event rows 读取；
- legacy Jul30 response rows 读取；
- private/order endpoint；
- model/score/actionability；
- own-order/fill/PnL artifact。

### 12.5 Independent QA 发挥了真正作用

QA 没有把 business positive tests 当作最终事实源。它发现了真实 causal、
semantic、attestation、durability 和 package-trust defects，并持续阻止
Stage 5 消费一个不完整的 dependency。

## 13. 已经超前完成的部分

Stage 4 最终不只是 Episode builder，它还完成了：

- 全部 `38,403,836` feature rows 的 source-semantic replay；
- 全部 Family A/B views 的 exact replay；
- anchors、sparse、grid、event-count、outcomes exact replay；
- current/archive executable/test byte identity；
- exact `12`-projection aggregate attestation；
- exact `28`-key aggregate count mapping；
- durable research inventory；
- exact filesystem type closure；
- atomic publication 与 zero-write；
- production coherent-rehash attack matrix。

这些工作实质上覆盖了 Ordered Queue Stage 5 的大部分验证任务。

但下面的研究仍未开始：

- case distance / retrieval；
- distribution baseline 和 model selection；
- Aug03/Aug04 transfer；
- Aug07 first-read；
- incremental cross-spread vs queue-shock；
- public latency/actionability；
- own-order lifecycle；
- randomized EV。

特别是，当前项目还没有：

- 把 offline detector 接入 production-equivalent monotonic/native hot path；
- 冻结真实 `KEEP/CANCEL_RISK_SIDE` decision contract；
- 验证 own-order lifecycle；
- 验证 randomized EV。

## 14. Stage 5 前必须修改的实施方法

### 14.1 新增 load-bearing surface matrix

每个 stage 开工前必须冻结：

| Output surface | Authoritative source | Decision time | Exact fields | Rebuild oracle | Negative mutation | Durable evidence |
| --- | --- | --- | --- | --- | --- | --- |
| anchors | Stage 3 candidate truth | Candidate/Confirm | all fields | source stream | relabel/quality | count + digest |
| features A/B | immutable events + contract | landmark | all fields | full row stream | value/provenance/version | digest + mismatch |
| outcomes | immutable public events | post-Candidate | all fields | interval replay | markout/interval/censor | digest + audit |
| manifest evidence | source-owned contract | publication | exact keys/types/values | attestation validator | missing/extra/type | canonical JSON |
| package tree | filesystem `lstat` | admission | path/type/mode/bytes | exact tree scan | file/dir/symlink/special | full inventory |

任何 output family 没有对应 negative test，都不能进入 `待验收`。

### 14.2 将 package identity 分层

至少拆为：

```text
research_data_identity
runtime_contract_identity
publication_envelope_identity
```

trust-only repair 应能保持 research-data layer 不变，只重建 runtime/
contract/envelope。最终 QA 仍然可以做一次 full rebuild。

### 14.3 Full build 前先跑 hostile preflight

第一轮 1.56GB build 之前，fast fixture 必须覆盖：

- wrong generating source event；
- detector burst termination；
- Candidate left truncation；
- 每个 artifact family coherent mutation；
- partial feature/view projection；
- evidence missing/extra/type/value；
- manifest missing/extra keys；
- unknown file/directory；
- dangling/file/directory symlink；
- FIFO/socket；
- root symlink；
- runtime/archive divergence；
- partial publication；
- default-Python zero-write。

五轮后续 finding 中的大部分都可以在 full build 前暴露。

### 14.4 Producer kernel 与 independent oracle 同时存在

复用 pure row constructor 可以减少 producer/verifier drift，但 producer
和 verifier 也可能共享同一个 bug。

后续 stage 应同时具备：

- source-owned pure contract/kernel；
- 独立实现的 audit projection；
- production coherent-rehash attacks；
- session-level aggregate reconciliation。

### 14.5 Evidence durability 成为硬 gate

任何：

```text
unchanged from previous repair
```

都必须在下一次编辑前先持久化 prior inventory。`/tmp` 可以用于执行，
不能作为跨轮次最终证据。

## 15. Stage 5 应改成 gap audit

Stage 5 不应重新执行 Stage 4 QA 已经接受的全部工作。

建议先发布：

| Planned Stage 5 item | QA Round 6 已接受 | 剩余工作 |
| --- | --- | --- |
| anchors | 是 | 汇总 accepted evidence |
| all-candidate outcomes | 是 | 汇总 accepted evidence |
| per-feature observed time | 是 | 汇总 accepted evidence |
| interval censoring | 是 | 汇总 accepted evidence |
| exact paths | 是 | 汇总 accepted evidence |
| deterministic rebuild | 是 | 汇总 accepted evidence |
| dependence blocks | 部分 | 发布独立 block membership/validation conclusion |
| consolidated Gate A/B report | 未作为独立 artifact 发布 | 新建报告，不改 data |

下一任务应以 validation/report closure 为主，不应默认再次重写 Episode
producer。

## 16. 修订后的 Stage exit criteria

未来 stage 只有同时满足以下七项，才能进入 `待验收`：

1. 每个 load-bearing field 都有 authoritative source。
2. 每个 field 要么 exact source-derived，要么显式 unavailable。
3. 每个 evidence/manifest section 有 exact key/type/value universe。
4. 每个 filesystem entry 都进入 exact identity universe。
5. 每个 cross-repair unchanged claim 都有 durable prior evidence。
6. 每个 artifact family 至少有一个 coherent-rehash negative test。
7. fast hostile preflight 通过后，才运行 full deterministic build。

QA 应在其中任一项未定义时直接拒绝，即使全部 positive tests 通过。

## 17. 当前决定

截至 2026-08-17：

- `0815T003` 已通过并关闭；
- final Jul30 Episode v3 package 可以作为 auditable dependency；
- Stage 5 已满足 predecessor QA gate；
- Stage 5 没有派发；
- 等待用户审阅本复盘，并确认新的 stage/acceptance structure。

应当把最终状态理解为：

```text
Stage 4 在五次返修后，终于补齐了 acceptance architecture。

下一阶段应消费 accepted package，
不应在没有新证据的情况下重新打开它。
```

## 18. Evidence Index

Initial：

- `.workflow/reports/0815T003-business.md`
- `.workflow/reports/0815T003-qa.md`

Repair history：

- `.workflow/reports/0815T003-business-r1.md`
- `.workflow/reports/0815T003-qa-round2.md`
- `.workflow/reports/0815T003-business-r2.md`
- `.workflow/reports/0815T003-qa-round3.md`
- `.workflow/reports/0815T003-business-r3.md`
- `.workflow/reports/0815T003-qa-round4.md`
- `.workflow/reports/0815T003-business-r4.md`
- `.workflow/reports/0815T003-qa-round5.md`
- `.workflow/reports/0815T003-business-r5.md`
- `.workflow/reports/0815T003-qa-round6.md`

Current QA mirror：

- `docs/qa-acceptance-report.md`

Durable research inventory：

- `.workflow/reports/0815T003-round4-pre-repair-research-inventory.csv`

Controller plan：

- `docs/skhynix_trigger_aligned_episode_research_implementation_plan.md`
