# Research Package Trust Kernel 执行方案

日期：2026-08-17

修订：2026-08-20 review remediation；冻结 surface-matrix schema、
accepted-version registry/bootstrap、amdserver archive 前置事实、
hostile-first gate、cleanup 两阶段断言、identity reverse-binding 和成本边界。

状态：

- 已由用户于 `2026-08-20` 确认为 v2 的强制前置和下一个正式任务边界；
- 已于 `2026-08-20` 正式派发
  `0820T001 / RESEARCH-PACKAGE-TRUST-KERNEL-LAYERED-IDENTITY-AND-STAGE4-PARITY`；
- 当前状态为 `执行中`；
- review remediation 已新增机器 schema 和空 registry bootstrap；
- 派发时尚未抽取 kernel 代码、修改 Stage 4 package、清理目录或传输归档。

## 1. 目标

本任务把 SKHYNIX Stage 4 在 QA Round 3-6 中逐轮补齐的通用信任逻辑，
沉淀为一次性、版本化、可被后续 research stage 复用的工程基础设施。

交付目标有六项：

1. 建立 `Research Package Trust Kernel` 共享模块。
2. 将 package identity 拆成三个相互独立、可组合的身份层。
3. 用已验收 Stage 4 package 证明抽取前后 admission parity。
4. 把 surface matrix、七条 exit criteria 和 hostile preflight 固化进
   `.workflow`。
5. 删除 `local_live_analysis/` 下七个已经确认的 Stage 4 空残留目录。
6. 把 1.5GB Stage 4 正式包原样归档到 amdserver 的 durable archive
   root。

完成后的默认规则是：

```text
通用 package trust 由 accepted kernel version 负责。

每个新 research stage 只负责：
1. pin accepted kernel；
2. 填完整 surface matrix；
3. 实现并验证自己的 domain oracle / adapter；
4. 证明自己的 outputs 与 surface matrix 一致。
```

这不会取消 stage-specific QA。它只把下面这些与研究对象无关的信任问题
从每个 stage 的 QA 中移出：

- exact evidence object；
- exact key/type/value universe；
- canonical serialization；
- exact `lstat` tree closure；
- layered inventory identity；
- atomic publication；
- verify-only zero-write；
- fail-closed admission。

## 2. 已验收基线

本任务必须把以下 Stage 4 QA Round 6 结果作为 immutable parity anchor。

正式 package：

```text
local_live_analysis/
skhynix_trigger_aligned_episode_research_v1_stage04_jul30_episode_v3/
```

接受身份：

| Identity | Accepted value |
| --- | --- |
| files / artifacts / bytes | `107 / 106 / 1,561,307,420` |
| legacy core | `78be6559c7ac5aaec4d5411b042f7ddcce522473f60f987237bcaa9d2dd5e157` |
| legacy full inventory | `669fb7d12f25cfa7828aec0fb1546398b1def754952de2290cd19784a477a433` |
| frozen contract | `b80b9bae2d6cf18cb6e7be4f133467138527fec20eef7054ab38b7f8c70cefde` |
| manifest raw bytes | `2c802336c1446eaf50eaf5ef546b115046abe3a1b69fdd1fcaea31d22a8e64a6` |

Durable research-data identity：

| Metric | Accepted value |
| --- | --- |
| research files | `99` |
| research bytes | `1,560,514,934` |
| canonical inventory SHA256 | `bb5aed2099b1a97da5b476d4b09dfe4a7bac06f0bf331f8ca864a88daf5c9232` |
| durable inventory file SHA256 | `07521dbc2c8bfe5f41a3ab14ac5493591dad878d52053b2d74aad251b9f022df` |

QA Round 6：

| Evidence | Accepted value |
| --- | --- |
| QA status | `已通过` |
| P0/P1/P2/P3 | `0/0/0/0` |
| aggregate negative calls | `98/98` fail closed |
| direct tree calls | `36/36` fail closed |
| production package tree attacks | `12/12` fail closed |
| current related tests | `396 passed` |
| archived Stage 4 focused tests | `97 passed` |
| QA report SHA256 | `0b85ae527f7e91f40f3e4401969596ff6462964a7e05cb8d0de73e763ad3e592` |

本任务不能重新解释、覆盖或替换这些 accepted identities。新分层身份是
附加的 trust representation，不是对 legacy Stage 4 验收结果的改写。

## 3. 非目标和硬边界

本任务是纯工程任务，不做任何新的研究判断。

禁止：

- 修改 99 个 accepted research CSV/GZ；
- 重算或改变任何 Episode row；
- 改变 Candidate、Confirmed、Family A、Family B 定义；
- 改变 feature、path、outcome、censor 或 detector semantics；
- 重建 1.5GB Stage 4 package；
- 读取 Aug03/Aug04 future event rows；
- 读取 Aug07 event rows；
- 运行 case retrieval、model、score 或 transfer research；
- 运行 actionability、own-order、fill、PnL 或 live execution；
- 把 SKHYNIX symbol、字段、计数、segment ID 或绝对路径写进通用 kernel；
- 原地修改 accepted Stage 4 package；
- 删除任何非空目录；
- 在 remote archive 通过前删除本机正式包。

允许：

- 只读扫描 accepted Stage 4 package；
- 只读运行一次完整 Stage 4 admission；
- 构建秒级 synthetic fixtures；
- 新增共享 kernel、Stage 4 compatibility adapter、测试和 workflow 模板；
- 创建 package 外部的 layered trust envelope；
- 将正式包 byte-exact 复制到 amdserver；
- 删除七个 exact-name、真实目录、内容为空的残留目录。

## 4. 总体架构

### 4.1 Trust boundary 分层

```mermaid
flowchart LR
    S["Stage-specific producer"] --> D["Research data"]
    S --> O["Domain oracle"]
    M["Surface matrix"] --> A["Stage adapter"]
    O --> A
    D --> A
    K["Accepted Trust Kernel"] --> A
    A --> R["research_data_identity"]
    A --> C["runtime_contract_identity"]
    A --> E["publication_envelope_identity"]
    R --> P["composite package identity"]
    C --> P
    E --> P
```

责任边界：

| Component | Owns | Must not own |
| --- | --- | --- |
| producer | 研究数据生成 | 通用 package trust 规则 |
| domain oracle | source semantics 重建 | filesystem/publication policy |
| stage adapter | surface matrix 到 kernel contract 的映射 | 自创 kernel 规则 |
| trust kernel | evidence、tree、identity、publication、admission | SKHYNIX 或策略语义 |
| workflow | 派发与 QA 的结构化 gate | 研究结论 |

### 4.2 不能错误地“一次 QA 后永远信任”

Kernel QA 通过后，后续 stage 可以不再重复攻击 kernel 内部，但仍必须检查：

1. pin 的 kernel version、source-tree SHA 和 acceptance receipt 都 exact；
2. stage adapter 实际调用了被 pin 的 kernel；
3. surface matrix 完整覆盖当前 stage 的全部 load-bearing outputs；
4. domain oracle 对当前 stage 的字段、时间和 source truth 是正确的；
5. adapter 没有在 kernel 前后缩小、伪造或绕过 evidence；
6. package 的实际层级身份与 manifest 声明一致。

Kernel 只摊薄通用信任成本，不替代研究语义 QA。

## 5. Deliverable A：Research Package Trust Kernel

### 5.1 建议代码布局

第一版使用当前 repository 最稳定的 Python import 边界：

```text
examples/hyperliquid/research_package_trust/
    __init__.py
    errors.py
    canonical.py
    evidence.py
    tree.py
    identity.py
    publication.py
    admission.py
    contracts.py

examples/hyperliquid/research_package_trust_cli.py
examples/hyperliquid/research_package_trust_stage4_adapter.py

examples/hyperliquid/test_research_package_trust_evidence.py
examples/hyperliquid/test_research_package_trust_tree.py
examples/hyperliquid/test_research_package_trust_identity.py
examples/hyperliquid/test_research_package_trust_publication.py
examples/hyperliquid/test_research_package_trust_stage4_parity.py
```

选择这一位置的原因：

- 当前 Stage 1-4 research tooling 已集中在 `examples/hyperliquid/`；
- 不需要新增 packaging/install 步骤；
- existing scripts 可以稳定 import sibling package；
- 模块本身通过 purity contract 禁止依赖任何 SKHYNIX/cross-exchange
  producer。

未来如需跨 `examples/` 目录复用，可以在 kernel v2 中迁入正式 Python
package；不能在 v1 实现中同时扩大 packaging 范围。

### 5.2 Kernel public API

Kernel v1 至少提供以下稳定 API：

```text
scan_exact_tree(root, tree_contract)
build_inventory(root, surface_contract)
validate_inventory_against_surface(inventory, surface_contract)
validate_exact_object(observed, exact_contract)
validate_surface_matrix(matrix)
load_accepted_version_registry(registry_path, registry_schema)
get_accepted_version(registry, kernel_name, kernel_version)
validate_pinned_version(declared_pin, accepted_registry_entry)
compute_research_data_identity(...)
compute_runtime_contract_identity(...)
compute_publication_envelope_identity(...)
compute_composite_package_identity(...)
admit_package(package_root, contract, semantic_evidence)
publish_atomically(staging_root, final_root, publication_contract)
```

Public return values 必须是 dataclass 或 canonical JSON-compatible object。
失败必须抛出统一：

```text
TrustKernelError(code, location, detail)
```

测试和下游 QA 应优先断言稳定 error code，不依赖易变的完整英文错误文本。

### 5.3 Purity contract

Kernel source 必须满足：

- 不 import `cross_exchange_trigger_aligned_episodes`；
- 不 import任何 SKHYNIX-specific module；
- 不包含 `SKHYNIX`、`SKHX`、`Jul30`、Family A/B 字段或 accepted counts；
- 不包含 absolute package/input root；
- 不读取 network、environment secrets、clock 或当前日期；
- admission API 只读；
- publication API 只能写调用方明确传入的 staging/final roots；
- canonicalization、hash 和排序规则完全由显式 contract 决定；
- 相同 bytes/contract 在 macOS 与 Linux 产生相同 identity；
- `bool` 不得通过 canonical integer validation；
- unknown key、unknown entry type 和 unknown contract version 全部
  fail closed。

### 5.3.1 Kernel v1 symlink policy

Kernel v1 对 package root 和全部 descendants **绝对禁止 symlink**：

- dangling symlink、指向 package 内 file/dir 的相对 symlink、指向
  package 外的 symlink、root symlink 全部 fail closed；
- 不提供“点名允许内部相对 symlink”的例外；
- tree contract 的合法 entry type 只有真实 directory 和真实 regular
  file；
- future 如需支持 symlink，必须创建新 kernel major contract/version，
  不能在 v1 下扩大 allowlist。

Stage 4 accepted package 的 exact tree 已证明 symlink/special 为 `0`，因此
该政策不会改变 accepted package 语义。

### 5.4 从现有 Stage 4 抽取的通用逻辑

应抽取：

- canonical JSON bytes/hash；
- exact typed object validation；
- exact key-universe validation；
- canonical SHA256 validation；
- exact binding derivation；
- `lstat` entry classification；
- exact root/descendant tree scan；
- regular-file inventory；
- artifact file/directory closure；
- layered identity calculation；
- staging fsync；
- atomic directory publication；
- verify-only zero-write support；
- canonical admission result。

不得抽取到 kernel：

- `_candidate_iter()`；
- `_feature_rows()`；
- `_outcome_row()`；
- `_verify_source_semantics()` 的 SKHYNIX projection implementation；
- detector burst、Candidate/Confirmed、interval censoring 规则；
- 12 个 projection 的名称、字段和 accepted counts；
- Stage 1/2/3 absolute dependency roots；
- Stage 4 report 文本。

这些内容保留在 immutable legacy implementation 或 Stage 4 adapter。

### 5.5 Kernel version 和 acceptance pin

Kernel name/version 分开冻结：

```text
kernel_name = research_package_trust_kernel
kernel_version = v1
combined_display_id = research_package_trust_kernel_v1
```

验收后生成：

```text
baselines/research_package_trust_kernel/v1/
    v1_acceptance_package/
        execution_plan.md
        kernel_acceptance.json
        api_contract.json
        negative_matrix.json
        fixture_inventory.json
        stage4_parity.json
        qa_report.md
        acceptance_package_inventory.json
```

Registry 固定位置：

```text
baselines/research_package_trust_kernel/accepted_versions.json
```

Registry schema 固定位置：

```text
.workflow/workflow-kit/research-package-kernel-registry.schema.json
```

`.gitignore` 必须只对白名单路径开放：

```text
baselines/research_package_trust_kernel/**
```

其他历史 `baselines/*` 继续忽略，避免为了 registry 而意外纳入大体积
本地产物。Registry 和 future acceptance package 若被 git ignore，Gate 1
直接失败。

派发前 bootstrap registry 必须：

- schema version：
  `research_package_trust_kernel_accepted_versions_v1`；
- `registry_revision=0`；
- `versions=[]`；
- 空 registry 不表示任何 kernel 已接受；
- business、test、QA 线程对 accepted registry 只读。

Registry 维护权只属于总控。v1 bootstrap 流程固定为：

1. business 在 `.workflow/reports/<TASK_ID>-kernel-candidate/` 生成 candidate
   acceptance package，不写 accepted registry；
2. independent QA 只验证 candidate 并写 QA report，不自行把 candidate
   标为 accepted；
3. QA 结果为 `已通过` 后，总控在一个 closure commit 中：
   - 把 exact candidate bytes 发布到
     `baselines/research_package_trust_kernel/v1/v1_acceptance_package/`；
   - 计算 acceptance-package inventory；
   - 将唯一 v1 accepted entry 原子写入 registry；
   - `registry_revision: 0 -> 1`；
4. registry entry 与 acceptance package 任一 byte 不一致，closure
   fail closed，v1 仍不可 pin；
5. future task 只能 pin registry 中 `status=accepted` 的 entry，不能 pin
   candidate path 或 QA 聊天结论。

Registry semantic validator 还必须要求：

- canonical pretty JSON raw bytes；
- `(kernel_name, kernel_version)` exact unique；
- entries 按 `(kernel_name, numeric_version)` 排序；
- 只允许 `status=accepted`；
- `registry_revision` 每次 controller promotion exact `+1`；
- 已存在 accepted entry 不可原地覆盖、删除或改写；
- acceptance-package path、plan/QA/receipt/inventory SHA 全部可读取并
  exact；
- registry、schema 和 acceptance package 均为 git-trackable path，
  `git check-ignore` 不得将其排除。

Kernel bootstrap task 的 surface matrix 使用：

```json
{
  "mode": "bootstrap_candidate",
  "kernel_name": "research_package_trust_kernel",
  "candidate_version": "v1",
  "registry_path": "baselines/research_package_trust_kernel/accepted_versions.json",
  "registry_schema_path": ".workflow/workflow-kit/research-package-kernel-registry.schema.json"
}
```

每个 future accepted-kernel stage 必须在 task 文件和 machine matrix 中
同时 pin：

```text
kernel_name
kernel_version
registry_path
registry_entry_sha256
kernel_source_tree_sha256
kernel_api_contract_sha256
kernel_negative_matrix_sha256
kernel_qa_report_sha256
kernel_acceptance_task_id
```

Runtime 必须在 import stage adapter、读取 research package 或运行
domain oracle 前执行：

```text
load registry
-> validate registry schema
-> get exact (kernel_name, kernel_version)
-> canonical-hash registry entry
-> validate every declared pin field
-> only then load accepted kernel
```

task Markdown、surface-matrix JSON、runtime pin 三处任一不一致均 fail
closed。不得根据目录名、latest alias 或 import 成功推断 accepted version。

Version string 相同但任何 source byte 改变，都视为非法 drift。

任何 kernel 修改必须：

1. 使用新 version；
2. 新建正式 workflow task；
3. 重跑完整 kernel negative matrix；
4. 重新独立 QA；
5. 发布新的 acceptance receipt。

## 6. Deliverable B：三层 Package Identity

### 6.1 Identity 1：`research_data_identity`

它只描述研究数据本身。

Canonical inventory row：

```json
{
  "path": "relative/path.csv.gz",
  "bytes": 123,
  "sha256": "lowercase-64-hex"
}
```

计算规则：

- relative path 必须 canonical；
- path、bytes 和 SHA 全部进入 identity；
- absolute root、mtime、ctime 和 host 不进入 identity；
- symlink 或 special entry 在计算前 fail closed；
- 一个文件必须且只能属于一个 research-data surface；
- unknown、unassigned 或 duplicate assignment 全部 fail closed。

`surface_id`、schema 和 file-to-surface assignment 属于
`runtime_contract_identity`。这样 research-data identity 只回答“哪些
research bytes 被冻结”，不会因为解释合同改变而错误变化。

对 legacy Stage 4 compatibility adapter：

- research-data file set 必须 exact 等于 durable inventory 的 99 个文件；
- bytes 和 raw SHA 必须逐文件 exact；
- 新 identity 的 canonicalization 应复用 durable inventory 口径；
- 目标值必须等于：
  `bb5aed2099b1a97da5b476d4b09dfe4a7bac06f0bf331f8ca864a88daf5c9232`。

### 6.2 Identity 2：`runtime_contract_identity`

它描述“如何解释和验证 research data”。

必须绑定：

- accepted kernel pin；
- stage adapter source SHA；
- domain oracle source SHA；
- surface matrix SHA；
- 每个 research file 的 exact surface assignment；
- schema/field contracts；
- exact evidence contract；
- dependency identities；
- input-binding contract；
- calculation/contract versions；
- packaged runtime tests 的 exact source SHA；
- hard-boundary declaration。

它不重新哈希 1.56GB research data，只绑定
`research_data_identity` 的 expected relation。

### 6.3 Identity 3：`publication_envelope_identity`

它描述 package 如何被发布、定位和承认。

必须绑定：

- `research_data_identity`；
- `runtime_contract_identity`；
- exact envelope file/directory universe；
- tree-entry type contract；
- canonical manifest bytes；
- atomic publication contract；
- verify-only zero-write contract；
- archive/reference metadata schema；
- composite package identity rule。

为了避免 manifest 自引用：

- envelope inventory 不包含最终 `package_seal.json` 自身；
- `package_seal.json` 保存 R/C/E 三个 identity 和 composite identity；
- admission 对 seal 使用 exact canonical bytes；
- exact tree contract 明确 seal 是唯一允许但不参与自身摘要的文件。

### 6.4 Composite identity

统一消费入口使用：

```text
package_identity = SHA256(
  canonical_json({
    "research_data_identity": R,
    "runtime_contract_identity": C,
    "publication_envelope_identity": E
  })
)
```

### 6.5 Identity 隔离验收

必须用 metamorphic tests 证明：

| Mutation | R | C | E | Composite |
| --- | --- | --- | --- | --- |
| research data byte/path | change | valid reseal requires change; old C rejected | valid reseal requires change; old E rejected | change |
| schema/surface assignment | unchanged | change | change | change |
| kernel/adapter/contract/test | unchanged | change | change | change |
| envelope/report/publication policy | unchanged | unchanged | change | change |
| absolute package relocation | unchanged | unchanged | unchanged | unchanged |
| mtime/ctime only | unchanged | unchanged | unchanged | unchanged |
| symlink/special entry | reject | reject | reject | reject |
| unknown file or directory | reject | reject | reject | reject |

这张 mutation matrix 是 identity layering 的硬验收，不是说明性示例。

Research-data reverse binding 必须单独执行以下 metamorphic test：

```text
mutate exactly one byte in one research CSV/GZ
-> observed R' != accepted R
-> reuse old C and old E
-> admission fails on C.expected_research_data_identity
   before any trusted composite identity is returned
-> recompute only R while retaining old C/E
-> admission still fails
-> a valid republish requires new C binding R',
   new E binding R'/C', and a new composite identity
```

因此“R、C、E 分层”不表示 R 可以独立替换。分层只允许判断哪一层首先
发生变化；一个可接受的 composite 始终要求 C/E 对当前 R 的 exact
反向绑定。

### 6.6 Legacy Stage 4 迁移方式

禁止原地迁移 accepted Stage 4 package。

本任务在 package 外创建小型 compatibility envelope，例如：

```text
local_live_analysis/
skhynix_stage04_episode_v3_trust_envelope_v1/
    surface_matrix.json
    research_data_inventory.json
    runtime_contract_inventory.json
    publication_envelope_manifest.json
    package_seal.json
    stage4_legacy_identity_bridge.json
```

Bridge 必须记录：

- legacy core/full/contract/manifest identities；
- 99-file durable research identity；
- accepted QA Round 6 report identity；
- legacy package locator 只作为运行参数，不进入三层 identity；
- accepted package root relocation 不改变 identity；
- legacy package 本身没有被修改。

Future native package 可以使用 content-addressed research-data object 和
small versioned envelope；legacy Stage 4 只通过 bridge 进入新体系。

## 7. Deliverable C：Stage 4 Admission Parity

### 7.1 Parity 原则

复用 Stage 3 detector extraction 的纪律：

```text
抽取不是“看起来等价”。

抽取后的 kernel + Stage 4 adapter 必须在机器上复现
已验收实现的同一判定、同一 identity 和同一 fail-closed boundary。
```

### 7.2 Legacy implementation 保持冻结

accepted package 内的：

- archived builder；
- archived admission；
- archived focused tests；
- frozen contract；
- manifest；

保持 byte-exact，不原地改造。

新 kernel 通过 compatibility adapter 读取 legacy package。这样能够：

- 保留 Round 6 accepted evidence；
- 避免 current source 改动导致 archived-source binding drift；
- 不要求重新构建 1.5GB package；
- 让 legacy verifier 成为 parity oracle，而不是被抽取过程覆盖。

### 7.3 Full admission parity

业务线程和独立 QA 各运行一次完整、只读 admission：

```text
accepted legacy verifier -> accepted Stage 4 package
new kernel + Stage 4 adapter -> same accepted Stage 4 package
```

两侧必须逐项一致：

- pass/fail；
- files/artifacts/bytes；
- legacy core/full/contract/manifest；
- 99-file durable inventory；
- 12 projection names；
- 12 projection fields；
- expected/observed row counts；
- expected/observed digest；
- mismatch rows；
- 28 aggregate-output counts；
- exact path/directory universe；
- boundary declarations；
- dependency/input identities；
- zero-write result。

新 kernel 额外发布 R/C/E/composite identities，但不能改变 legacy result。

### 7.4 不做 full rebuild

本任务不运行：

```text
cross_exchange_trigger_aligned_episodes.py --output-dir <new 1.5GB package>
```

原因：

- 研究数据生成器没有改变；
- parity 目标是 admission extraction；
- durable inventory 已证明 99 个 research files 不变；
- full rebuild 会重新引入本任务要消灭的成本结构。

如果 parity 只有重新 build 才能成立，应直接判定抽取设计失败，不允许
用重建掩盖 compatibility 问题。

## 8. Deliverable D：永久 Negative Test Matrix

### 8.1 秒级 fixture

创建一个小型 package fixture：

- 2-4 个 research data files；
- 2 个 runtime/contract files；
- 1 个 canonical manifest；
- 1 个 surface matrix；
- 1 个 package seal；
- 小于 `1MB`；
- 完整测试目标应在普通开发机秒级完成。

Fixture 只缩小数据体积，不能缩小 contract shape 和 attack classes。

### 8.2 Aggregate matrix

保留 QA Round 6 的完整逻辑：

- `49` 个 unique mutation cases；
- current kernel 与 frozen packaged kernel snapshot 各执行一次；
- 总计 `98/98` negative calls。

必须覆盖：

- semantic evidence missing/not-dict/false/extra；
- scope missing/extra/type/value；
- projection missing/extra/non-dict；
- entry missing/extra/non-dict；
- fields type/value/order；
- count value/bool/string；
- digest mismatch/malformed/uppercase/non-string；
- mismatch nonzero/bool/string；
- exact counts missing/extra/value/type；
- aggregate counts missing/extra/value/type；
- binding duplicate/empty/missing/extra/bool/string。

### 8.3 Direct tree matrix

保留同一调用口径：

- 6 类 entry/root attack；
- 3 个 load-bearing kernel boundary；
- current/frozen 两套 kernel；
- 总计 `36/36` calls。

Attack：

- dangling symlink；
- symlink -> regular file；
- symlink -> directory；
- FIFO；
- Unix socket；
- root symlink。

Boundaries：

- exact tree inventory；
- artifact record construction；
- artifact/tree closure。

### 8.4 Production-shape admission matrix

使用完整 CLI/admission path，但仍使用小 fixture：

- 6 类 tree attack；
- current/frozen 两套 kernel；
- 总计 `12/12` calls。

必须证明：

- `rc != 0`；
- `verified=false`；
- 不输出可信 R/C/E/composite identity；
- 在 domain semantic replay 前快速拒绝；
- 不产生 package 内写入。

### 8.5 额外永久测试

新增：

- identity layer isolation matrix；
- canonical relocation；
- seal self-reference exclusion；
- surface assignment missing/duplicate；
- unknown kernel version；
- kernel source pin drift；
- adapter bypass；
- zero-write under default Python；
- partial publication；
- existing final output preservation；
- staging cleanup；
- source and frozen-kernel result parity。

## 9. Deliverable E：Workflow 模板固化

### 9.1 适用范围

不是所有普通代码任务都需要 surface matrix。

以下任务必须使用新模板：

```text
task_type = research_package | research_package_infrastructure
produces_research_package = true
```

包括：

- 新研究数据集；
- 新 label/feature/outcome package；
- 新 model-evaluation package；
- 会被后续 stage 当作 accepted dependency 的 research artifact bundle。

普通 bugfix、docs-only 和不产生 research package 的任务可以声明：

```text
task_type = general
produces_research_package = false
```

任务类型只由 task 文件中的上述显式字段决定。Validator 禁止根据文件路径、
标题关键词、output 目录名或代码 import 猜测任务类型：

- `task_type=research_package` 或 `research_package_infrastructure` 时，
  `produces_research_package` 必须为 `true`，surface matrix 强制存在；
- `task_type=general` 时，`produces_research_package` 必须为 `false`；
- 交叉组合、缺字段或未知 enum 全部 fail closed；
- 本 Trust Kernel bootstrap task 使用
  `task_type=research_package_infrastructure`。

### 9.2 Workflow 文件改动

正式实现应修改或新增：

```text
.workflow/workflow-kit/workflow-manual.md
.workflow/workflow-kit/task-dispatch-template.md
.workflow/workflow-kit/thread-report-template.md
.workflow/workflow-kit/qa-acceptance-template.md
.workflow/workflow-kit/research-package-task-template.md
.workflow/workflow-kit/research-package-surface-matrix.schema.json
.workflow/workflow-kit/research-package-kernel-registry.schema.json
.workflow/workflow-kit/validate_research_package_task.py
baselines/research_package_trust_kernel/accepted_versions.json
AGENTS.md
```

### 9.3 派发文档强制字段

每个 research-package task 必须内嵌填好的表：

| Surface | Authoritative source | Decision/as-of time | Exact fields/keys | Rebuild oracle | Negative mutation | Durable evidence | Identity layer |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `<filled>` | `<filled>` | `<filled>` | `<filled>` | `<filled>` | `<filled>` | `<filled>` | `R/C/E` |

规则：

- 不允许空单元格；
- 不允许 `TBD`；
- 不允许只写 `same as source`；
- `unavailable` 必须说明原因和 fail-closed behavior；
- 每个 output family 至少一行；
- 每个 artifact 必须属于一个 surface；
- 每个 surface 必须绑定一个 identity layer；
- 每个 surface 必须有至少一个 negative mutation。

任务文件同时引用一份 canonical JSON：

```text
.workflow/contracts/<TASK_ID>-surface-matrix.json
```

JSON 是机器事实源；Markdown 表是派发时的人类审阅面。业务回报必须记录
最终 JSON SHA。

### 9.3.1 Surface Matrix JSON Schema

派发前冻结的 machine schema：

```text
path:
.workflow/workflow-kit/research-package-surface-matrix.schema.json

schema_version:
research_package_surface_matrix_v1

sha256:
fa6e10608855d7ff781be8ffd8d64b713244fe81db30ed70b8502e40244261d8
```

Accepted-version registry schema：

```text
path:
.workflow/workflow-kit/research-package-kernel-registry.schema.json

schema_version:
research_package_trust_kernel_accepted_versions_v1

sha256:
72ba717c6ba1a9fe288c36174c05d1d23ef37db45213f6010f9446d5fba8197b
```

Empty bootstrap registry：

```text
path:
baselines/research_package_trust_kernel/accepted_versions.json

sha256:
d4e045a5aeca78288ce38d66497ace3baca1962058c583adfc33ac5644b0d285
```

Surface schema 完整冻结：

- task ID、显式 task type 和 `produces_research_package=true`；
- bootstrap/accepted 两种 kernel pin；
- artifact relative path、entry type 和 required flag；
- authoritative source type、locator 和 identity；
- decision-time kind、field、relation、clock 和 observed-at rule；
- availability state；
- exact key/field universe、canonicalization、schema binding 和 row
  identity；
- rebuild oracle；
- 每个 surface 至少一个 negative mutation 及 expected stable error
  code；
- durable evidence；
- `R/C/E` identity layer；
- surface dependencies；
- 七条 exit-criteria planning records。

合法 `unavailable.reason` 词表冻结为：

```text
source_absent
source_not_collected
source_not_authoritative
not_identified
outside_task_scope
unsupported_by_input
dependency_not_accepted
```

`not_applicable` 只能使用：

```text
not_applicable_to_surface
```

不得使用自由文本 unavailable reason 绕过 schema。自由文本解释只能放在
report，不进入 machine enum。

JSON Schema 负责字段类型、必填性、枚举和局部 shape。
`validate_research_package_task.py` 还必须做 JSON Schema 无法完整表达的
cross-object semantic checks：

1. `surface_id` exact unique；
2. artifact path 全局 exact unique，且每个 artifact 恰好属于一个 surface；
3. `depends_on_surfaces` 全部存在且无 cycle；
4. negative `mutation_id` 全局 unique；
5. EC1-EC7 各出现且只出现一次；
6. Markdown surface IDs 与 JSON exact；
7. accepted kernel pin 的每个字段与 registry entry exact；
8. bootstrap pin 只允许
   `task_type=research_package_infrastructure`；
9. relative path canonical，禁止 absolute、`..`、空段和 separator
   normalization alias；
10. schema/registry raw bytes 与本节冻结 SHA exact。

Decision-time 示例必须覆盖不同语义，不能把所有 surface 写成
`fixed_origin`：

| Surface example | kind | field | relation | clock |
| --- | --- | --- | --- | --- |
| frozen research row | `fixed_origin` | `candidate_ts_ns` | `lte` | `receive_time_ns` |
| rolling risk state | `rolling_update` | `grid_endpoint_ns` | `lte` | `receive_time_ns` |
| package seal/report | `final_publication` | `null` | `not_applicable` | `publication_time` |

Kernel bootstrap task 的 schema/registry files 本身属于 C surface；空
registry 属于 E/acceptance-governance surface，不能误归到 R。

### 9.4 七条 Stage exit criteria

模板中固定加入：

1. 每个 load-bearing field 都有 authoritative source。
2. 每个 field 要么 exact source-derived，要么显式 unavailable。
3. 每个 evidence/manifest section 有 exact key/type/value universe。
4. 每个 filesystem entry 都进入 exact identity universe。
5. 每个 cross-repair unchanged claim 都有 durable prior evidence。
6. 每个 artifact family 至少有一个 coherent-rehash negative test。
7. fast hostile preflight 通过后，才允许 full deterministic build。

业务线程进入 `待验收` 前必须逐项写出：

```text
pass / fail / not-applicable-with-reason
```

### 9.5 Hostile preflight

research-package task 默认顺序：

```text
surface matrix validation
-> seconds-level hostile fixture
-> identity metamorphic tests
-> stage-specific small fixture
-> full build or full admission
-> business handoff
```

顺序由 machine receipt 强制，而不只靠业务报告叙述：

```text
.workflow/reports/<TASK_ID>-hostile-preflight.json
```

Receipt 必须记录：

- schema/version；
- task ID；
- current kernel source-tree SHA；
- frozen kernel snapshot SHA；
- surface schema/matrix SHA；
- registry schema/bootstrap SHA；
- fixture inventory SHA；
- aggregate/direct-tree/production-shape/metamorphic counts；
- stable error-code summary；
- `started_at_utc` / `completed_at_utc`；
- `passed=true`；
- canonical receipt SHA。

Full admission runner 在开始前必须：

1. 读取 receipt canonical bytes；
2. 重算 receipt SHA；
3. exact 比较当前 source/schema/fixture/matrix hashes；
4. 要求所有 negative counts 完整且 fail-open 为 `0`；
5. 原子写
   `.workflow/reports/<TASK_ID>-first-full-admission-start.json`，
   绑定 hostile receipt SHA。

缺 receipt、receipt drift、receipt 在 full-admission start 之后生成或任一
hash 不一致时，runner 必须拒绝启动 full admission。QA Gate 2 独立比较：

```text
hostile_preflight.completed_at_utc
<
first_full_admission.started_at_utc
```

并核对两份 receipt 的 hash binding。时间顺序只是辅助证据，真正 gate 是
内容寻址 receipt；修改系统时钟不能绕过 hash gate。

如果前三步任一步失败：

- 不允许运行 full build；
- 不允许发布 provisional package；
- 任务保持 `执行中` 或报告 `阻塞`。

### 9.6 QA Gate 0

QA 模板的第一项固定为：

```text
Gate 0: Surface Matrix Completeness
```

QA 必须在执行大型测试前检查：

- task 声明是否产生 research package；
- matrix 是否存在；
- machine schema 是否通过；
- Markdown 与 JSON 的 surface IDs 是否一致；
- 是否覆盖所有 artifact/output families；
- 七条 exit criteria 是否全部有证据；
- kernel pin 是否指向 accepted version。

任何一项未定义：

```text
直接未通过。

正向测试全部通过也不能覆盖 Gate 0 失败。
```

### 9.7 历史任务兼容

新规则只强制作用于生效日期之后新派发的 research-package task。

已有历史 task：

- 不批量改写；
- 不因缺少新字段而 retroactively 变成未通过；
- 如被重新打开或形成新 package version，则必须使用新模板。

## 10. Deliverable F：Stage 4 Hygiene 和 Durable Archive

### 10.1 七个空目录

只允许删除以下 exact paths：

```text
local_live_analysis/skhynix_trigger_aligned_episode_research_v1_stage04_jul30_episode_v3 2
local_live_analysis/skhynix_trigger_aligned_episode_research_v1_stage04_jul30_episode_v3 3
local_live_analysis/skhynix_trigger_aligned_episode_research_v1_stage04_jul30_episode_v3 4
local_live_analysis/skhynix_trigger_aligned_episode_research_v1_stage04_jul30_episode_v3 5
local_live_analysis/skhynix_trigger_aligned_episode_research_v1_stage04_jul30_episode_v3 6
local_live_analysis/skhynix_trigger_aligned_episode_research_v1_stage04_jul30_episode_v3 7
local_live_analysis/skhynix_trigger_aligned_episode_research_v1_stage04_jul30_episode_v3 8
```

删除前每个 path 必须：

- `lstat` 为真实 directory；
- 不是 symlink；
- exact entry count 为 `0`；
- path 等于 frozen allowlist；
- 不与正式 package path 相同。

任意一个 path 不满足：

- 七个目录全部不删除；
- 记录 fail-closed cleanup result；
- 不允许模糊匹配或扩大到其他 `stage04*` 路径。

Pre-delete 使用两阶段、全体先验检查：

```text
Phase A: read-only preflight over all seven
  -> exact raw path text equals frozen allowlist entry, including spaces
  -> parent root is the frozen local_live_analysis root
  -> lstat type=directory and not symlink
  -> capture st_dev/st_ino/st_mode/st_nlink
  -> os.scandir returns []
  -> no open task-owned writer/transfer process targets the path

only when all seven pass:

Phase B: delete
  -> acquire task-scoped cleanup lock
  -> re-lstat all seven and compare captured st_dev/st_ino/type
  -> readdir all seven again and require []
  -> call os.rmdir on each exact path
  -> never call rm -rf, glob, find -delete or recursive deletion
```

Phase A 任一失败时，filesystem mutation count 必须为 `0`。Phase B
recheck 任一失败时，在第一次 `os.rmdir` 前整体退出，七个目录全部保留。
`os.rmdir` 自身若因并发写入返回非空或 identity drift，任务失败并保留完整
error/evidence；不得改用递归删除补救。

删除后发布：

```text
.workflow/reports/<TASK_ID>-stage4-empty-dir-cleanup.json
```

内容包括 allowlist、pre-lstat、empty proof、post-absence 和执行时间。

### 10.2 amdserver durable root

`2026-08-20` 总控只读 preflight 已确认：

- SSH alias：`amdserver`；
- execution user/home：`molly / /home/molly`；
- git worktree root：`/home/molly/project/hftbacktest`；
- worktree parent：`/home/molly/project`；
- `/home/molly/project` 是真实 directory、非 symlink、`molly` 可写；
- `/home/molly/project/durable_archives` 当前不存在，但其 parent 可创建；
- planned final 和 `.tmp` path 当前都不存在；
- filesystem available bytes：`1,009,309,990,912`；
- package 三倍预算：`4,683,922,260` bytes；
- remote Python：`3.10.12`；
- `/`、`/home`、`/home/molly`、`/home/molly/project` 均为真实 directory，
  非 symlink。

该 preflight 没有创建目录、传输文件或修改 remote 状态。

冻结 canonical archive root：

```text
/home/molly/project/durable_archives/
skhynix_episode_research_v1/
stage04_jul30_episode_v3/
669fb7d12f25cfa7828aec0fb1546398b1def754952de2290cd19784a477a433/
```

该路径是 worktree sibling tree，不位于
`/home/molly/project/hftbacktest` 内。正式任务只能由 remote user
`molly` 创建。若执行时 user、parent identity、write permission、free
space、Python runtime、final/temp absence 或任一 parent entry type 与上述
freeze 不一致，archive phase 必须 `阻塞`，不得回退到 worktree 内路径或
另选未冻结目录。

布局：

```text
<archive-root>/
    package/
    trust_envelope/
    evidence/
        0815T003-qa-round6.md
        durable_research_inventory.csv
        archive_receipt.json
        source_inventory.json
        destination_inventory.json
```

`archive_receipt.json` 和其他 archive evidence 位于 package 外，不能改变
accepted package identity。

### 10.3 Transfer protocol

1. 本机用 new kernel exact `lstat` scanner 扫描正式 package。
2. 确认 source identities 等于 QA Round 6 anchors。
3. amdserver preflight：
   - canonical parent 是真实 directory；
   - final 和 temp 不存在；
   - 可用空间至少为 package bytes 的三倍；
   - 没有 symlink/special parent component；
   - Python runtime 可运行 kernel inventory CLI。
4. 由 `molly` 创建缺失的 exact durable parent components；每创建一级都
   立即 `lstat` 为真实 directory、拒绝 symlink/special，并 fsync parent。
5. 传输到同 filesystem 的 hidden temp root。
6. 在 amdserver 用 kernel 重算 exact tree、R/C/E 和 legacy identities。
7. source/destination 比较 exact relative path、entry type、mode、bytes、
   raw SHA。
8. fsync temp tree 和 parent。
9. atomic rename 为 final。
10. 在 final 上再运行一次 destination inventory。
11. 在 package 外原子写入 archive receipt。

Transfer script 必须前台同步完成：

- shell 使用 `set -euo pipefail`；
- `rsync`/copy/hash/inventory 子进程不得使用 `&`、`nohup`、`disown`、
  remote background shell 或 detached service；
- 如实现使用多个前台 child process，主脚本必须记录 PID 并逐个 `wait`；
- 所有 child 成功退出后才允许 rename/write receipt；
- receipt 写入 `transfer_process_mode=foreground_waited` 和完整 child exit
  status；
- business handoff 前，本机和 amdserver 的 task PID ledger 都必须为空。

禁止：

- 直接覆盖已有 final；
- 用 symlink 指向 worktree package；
- 只比较总 bytes；
- 使用 `is_file()/is_dir()` 形成 inventory；
- 未验证就删除 temp/source；
- 把 receipt 写进 accepted package。

### 10.4 Archive acceptance

Remote archive 必须满足：

- `107` regular files；
- `21` descendant directories；
- symlink/special `0`；
- bytes `1,561,307,420`；
- legacy core/full/contract/manifest exact；
- 99-file durable research inventory exact；
- R/C/E/composite exact；
- package tree 与本机 source 逐项 exact；
- remote archive path 不在任何 git worktree 内；
- transfer temp/residual partial path 为 `0`。

### 10.5 Replay portability 声明

当前 legacy Stage 4 verifier 还依赖 accepted Stage 1/2/3 和 Jul30 input
bindings。仅归档 1.5GB package 不等于归档全部 source-replay inputs。

Archive receipt 必须明确：

```json
{
  "schema_version": "research_package_stage4_archive_receipt_v1",
  "task_id": "<TASK_ID>",
  "source_host": "macmini",
  "destination_host": "amdserver",
  "destination_user": "molly",
  "archive_root": "/home/molly/project/durable_archives/skhynix_episode_research_v1/stage04_jul30_episode_v3/669fb7d12f25cfa7828aec0fb1546398b1def754952de2290cd19784a477a433",
  "transfer_process_mode": "foreground_waited",
  "all_child_exit_status_zero": true,
  "source_inventory_sha256": "<lowercase-64-hex>",
  "destination_inventory_sha256": "<same-lowercase-64-hex>",
  "legacy_core_sha256": "78be6559c7ac5aaec4d5411b042f7ddcce522473f60f987237bcaa9d2dd5e157",
  "legacy_full_inventory_sha256": "669fb7d12f25cfa7828aec0fb1546398b1def754952de2290cd19784a477a433",
  "research_data_identity": "bb5aed2099b1a97da5b476d4b09dfe4a7bac06f0bf331f8ca864a88daf5c9232",
  "runtime_contract_identity": "<lowercase-64-hex>",
  "publication_envelope_identity": "<lowercase-64-hex>",
  "composite_package_identity": "<lowercase-64-hex>",
  "byte_exact_package_archive": true,
  "kernel_trust_admission_portable": true,
  "full_source_semantic_replay_portable": false,
  "portability_class": "byte_exact_and_kernel_admission_only",
  "portability_statement": "This archive is a byte-exact copy of the accepted Stage 4 package and is portable for kernel/package admission. It is not a self-contained source-semantic replay archive. Full replay still requires the exact external Stage 1/2/3 packages and Jul30 inputs listed in external_dependency_bindings.",
  "external_dependency_bindings": [
    {
      "dependency_id": "<frozen-id>",
      "accepted_identity": "<lowercase-64-hex>",
      "source_locator": "<frozen-local-absolute-path>",
      "archived_with_this_task": false
    }
  ],
  "temp_paths_remaining": 0,
  "task_processes_remaining": 0,
  "started_at_utc": "<RFC3339>",
  "completed_at_utc": "<RFC3339>"
}
```

本任务不归档完整 Stage 1/2/3 和 Jul30 source inputs，因此
`full_source_semantic_replay_portable` 的 accepted value 固定为 `false`，
`portability_class` 和 `portability_statement` 必须使用上面的 exact
canonical value。即使 amdserver 恰好存在某些同名路径，也不能在本任务中
升级为 `true`。

remote verification 只要求：

- byte-exact package/tree identity；
- kernel/package admission；
- legacy package-level identities；
- external dependency bindings 的声明完整性。

它不尝试打开 `/Users/liu/Documents/...`。该本机路径在 amdserver 不存在或
不同不会使 byte-exact transfer 失败，而是由固定 `false` portability
声明诚实界定。Parity full source-semantic admission 仍在现有 accepted
local inputs 上完成。

本任务不静默扩大为“归档所有原始数据和所有前置 stage”。

### 10.6 本机 package 处置

本任务只把 amdserver 设为 canonical durable archive，不在同一任务中删除
本机正式 package。

原因：

- Stage 4 parity 需要该 package；
- 当前 accepted dependency paths 是本机 absolute roots；
- v2 Stage H0 尚未正式派发，其只读消费位置仍由后续任务冻结；
- 删除本机 package 会把纯 trust task 扩大成 dependency relocation。

任务通过后，本机目录降级为 cache。后续如需释放空间，必须另开一个明确
的 cache cleanup/relocation task。

## 11. 实施阶段

### Phase 0：冻结任务输入

交付：

- accepted baseline manifest；
- QA Round 6 identity anchors；
- accepted package pre-task exact inventory；
- 七个 empty-dir allowlist；
- frozen surface-matrix schema/path/SHA；
- frozen accepted-version registry schema/path/SHA；
- empty registry bootstrap path/SHA；
- confirmed amdserver user/root/parent/write/free-space/runtime facts；
- frozen amdserver archive root；
- task surface matrix。

Gate：

- 任何 accepted identity 不匹配，任务立即阻塞；
- 不进入 kernel extraction。

### Phase 1：实现 kernel 和小 fixture

交付：

- versioned pure kernel；
- stable error codes；
- synthetic package fixture；
- current/frozen kernel snapshot；
- 98/36/12 negative matrix；
- hostile-preflight canonical receipt；
- zero-write 和 atomic publication tests。

Gate：

- 所有测试必须在秒级 fixture 上通过；
- 不读取正式 package。
- hostile receipt 未通过或与 source/schema/fixture/matrix drift 时，
  full Stage 4 admission runner 必须拒绝启动。

### Phase 2：实现三层 identity

交付：

- R/C/E/composite contracts；
- identity manifests；
- mutation isolation matrix；
- Stage 4 layer assignment contract；
- legacy identity bridge。

Gate：

- 107 个 legacy files 必须全部且只属于一个 layer/surface；
- 99-file R identity 必须等于 durable accepted value。

### Phase 3：Stage 4 compatibility adapter 和 parity

交付：

- Stage 4 adapter；
- legacy/new verdict comparison；
- 一次 business full admission；
- no-write inventory comparison；
- parity report。

Gate：

- 任意 legacy field、count、digest、verdict 或 identity 不一致即失败；
- 不允许通过重建 package 修复。

### Phase 4：Workflow 固化

交付：

- manual/template/schema/validator 更新；
- valid/invalid task fixtures；
- Gate 0 测试；
- historical-task compatibility 测试。

Gate：

- 缺 matrix 的新 research-package task 必须被 validator/QA 模板拒绝；
- non-research task 不被误伤。

### Phase 5：Hygiene 和 remote archive

执行顺序：

1. archive preflight；
2. transfer temp；
3. remote exact verification；
4. atomic archive publish；
5. archive receipt；
6. 七个 empty dirs fail-closed cleanup。

Gate：

- archive 未完成时，不执行 cleanup；
- cleanup 不触碰正式 package；
- remote final 不允许部分成功。

### Phase 6：Business handoff

业务报告至少包含：

- commit；
- kernel pin；
- public API；
- negative matrix summary；
- identity isolation summary；
- Stage 4 parity result；
- pre/post accepted package inventory；
- workflow validator evidence；
- archive path/receipt/inventory；
- cleanup receipt；
- boundary scan；
- 是否进入 QA。

业务线程结束状态只能是：

```text
待验收
```

不能自行标记 kernel accepted。

## 12. Independent QA 方案

### Gate 0：Surface Matrix

第一项检查，失败直接 `未通过`：

- task matrix 完整；
- surface schema raw SHA exact 且 canonical JSON schema 通过；
- registry schema/bootstrap raw SHA exact；
- cross-object semantic validator 通过；
- artifact/surface/layer assignment exact；
- explicit task type 与 `produces_research_package` 组合合法；
- 七条 exit criteria 有证据；
- 无 `TBD` 或未解释 unavailable。

### Gate 1：Kernel purity 和 version pin

检查：

- source 中无 SKHYNIX/domain constants；
- import boundary；
- deterministic canonicalization；
- stable error codes；
- source/frozen snapshot identity；
- accepted-version registry schema；
- empty bootstrap registry exact；
- registry semantic validator、git trackability 和 immutable-entry
  behavior；
- business/QA 未写 accepted registry；
- candidate acceptance package 尚未被 future task pin；
- QA 通过后由总控 closure commit 发布 v1 acceptance package 和唯一
  accepted registry entry；
- task Markdown / matrix / runtime pin exact。

### Gate 2：Negative matrix

独立重跑：

- aggregate `98/98`；
- direct tree `36/36`；
- production-shape tree `12/12`；
- identity metamorphic matrix；
- zero-write；
- partial publication。

任意 fail-open 为 P1。

还必须检查 hostile-first 强制证据：

- hostile-preflight receipt canonical；
- receipt 绑定 current/frozen kernel、schema、fixture、matrix SHA；
- first-full-admission receipt 绑定 hostile receipt SHA；
- hostile completion 严格早于 first full admission start；
- full admission runner 在 missing/drift/failed receipt 下的拒绝测试通过。

### Gate 3：Stage 4 parity

独立运行一次完整 admission：

- legacy archived verifier；
- new kernel + Stage 4 adapter；
- accepted formal package。

要求所有 legacy results exact，并验证：

- accepted package pre/post byte、type、mode、mtime/ctime 不变；
- 99-file durable inventory不变；
- package 内无 pycache/write；
- 不创建新 Stage 4 package。

Bootstrap v1 本任务要求 business 和 independent QA 各运行一次完整只读
admission。这是 kernel 第一次建立 parity anchor 的一次性成本，不能用
未来 trust-only 快速路径绕过。

### Gate 4：Workflow enforcement

独立构造：

- 完整合法 task；
- 缺 surface；
- 缺 authoritative source；
- 缺 negative mutation；
- 缺 identity layer；
- 含 `TBD`；
- non-research task。

预期：

- 前者通过；
- 五个不完整 research tasks fail closed；
- non-research task 正常通过。

### Gate 5：Remote archive

独立读取 amdserver：

- execution user exact 为 `molly`；
- final path；
- final path exact 位于
  `/home/molly/project/durable_archives/` 且不在 git worktree 内；
- source/destination inventory；
- exact identities；
- receipt canonical bytes；
- temp/residual paths；
- package 是否位于 git worktree 外；
- replay portability 声明是否使用 frozen false/class/statement；
- transfer process mode 是否为 `foreground_waited`；
- child exit status 和 task PID ledger 是否全部闭合。

### Gate 6：Cleanup

确认：

- Phase A 在任何 mutation 前对七个 path 全部通过 exact
  path/lstat/inode/empty 检查；
- Phase B 在第一次 `os.rmdir` 前完成全体 recheck；
- 任一 Phase A/B precondition 失败 fixture 的 mutation count 为 `0`；
- 实现未调用递归、glob 或模糊删除；
- 七个 exact empty paths 不存在；
- 正式 package 存在且 identities 未变；
- 其他 `local_live_analysis` 路径未被删除；
- cleanup receipt exact。

### Gate 7：Hard boundary

确认：

- 99 个 research files unchanged；
- 未运行 full rebuild；
- 未改 research semantics；
- 未读取 later-session outcome；
- 未运行 model/actionability/order/live；
- transfer/test runner 未使用 detached/background mode；
- 本机和 amdserver task PID ledger 均为空；
- 没有后台 transfer/test 进程残留。

QA 只有全部 Gate 通过，才能将 kernel version 标记为 accepted。

## 13. 验收标准

任务完成必须同时满足：

1. Surface-matrix schema、registry schema 和 empty bootstrap registry
   已在派发前冻结并验证 raw SHA。
2. 通用 kernel 无 SKHYNIX specificity。
3. Kernel API、registry ownership、bootstrap promotion 和 future pin
   syntax 被 canonical contract 冻结。
4. 98/36/12 permanent negative matrix 在小 fixture 上通过。
5. hostile-first receipt 机器阻止 preflight 前 full admission。
6. R/C/E/composite identity 隔离和 R-change reverse-binding 测试通过。
7. Stage 4 legacy package 未修改、未重建。
8. 新 kernel 对 accepted Stage 4 package 的完整 admission 与 Round 6
   exact parity。
9. 99 research files exact，R identity 为 `bb5aed...9232`。
10. `.workflow` 对新 research-package task 强制 surface matrix、
   hostile preflight 和七条 exit criteria。
11. QA Gate 0 在 matrix/schema/registry 不完整时 fail closed。
12. 七个 exact empty dirs 经两阶段 precondition 安全删除。
13. 正式包在 frozen amdserver worktree 外路径形成 byte-exact durable
    archive。
14. archive receipt 固定声明 full source-semantic replay portability 为
    `false`。
15. transfer 前台完成且两端无 task process 残留。
16. 本机正式包保留为 cache，未在本任务中删除。
17. v1 execution plan、QA report、parity 和 acceptance inventory 一起
    进入 `v1_acceptance_package/`。
18. 独立 QA 为 `已通过`，且 P0-P3 全零。

## 14. 失败和回滚

### Kernel/Parity 失败

- legacy package 继续作为唯一 accepted dependency；
- 不发布 kernel acceptance receipt；
- future stage 不得 pin 未接受 kernel；
- 不修改 Stage 4 accepted status。

### Identity layering 失败

- 不允许只发布其中一两个 identity；
- R/C/E/composite 必须作为一个 versioned contract 同时进入 QA。

### Remote transfer 失败

- 保留本机正式 package；
- 删除或隔离 amdserver temp；
- 不创建 final；
- 不写 `passes=true` receipt。

### Cleanup precondition 失败

- 七个目录全部不删除；
- 记录失败 path 和 lstat/entry evidence；
- 不使用 `rm -rf stage04*` 一类宽匹配。

### Workflow validator 误伤历史任务

- 新规则只对新 task 或显式 reopened task 生效；
- 不批量修改历史任务；
- 不通过关闭 validator 来绕过问题。

## 15. 成本预算

Bootstrap v1 一次性目标预算：

| Work | Expected cost |
| --- | --- |
| kernel unit/negative tests | 秒级 |
| identity metamorphic tests | 秒级 |
| workflow validator tests | 秒级 |
| business full Stage 4 admission | 约 15-20 分钟 |
| QA full Stage 4 admission | 约 15-20 分钟 |
| full 1.5GB rebuild | `0` 次 |
| amdserver transfer | 约 1.5GB 单次传输 |

因此本次 bootstrap 预期有两次只读 full admission，总计约
`30-40` 分钟；这是 business 建立 parity evidence 和 independent QA
独立复核各一次，不是“一次 admission”的含混说法。

与 Stage 4 历史的比较有两个口径：

- 单轮返修口径：Stage 4 曾反复运行 Formal/Build A/Build B 和 QA fresh
  rebuild；本任务每侧只 admission，不 build；
- 五轮累计口径：Stage 4 每发现一个 trust assertion 缺口都重新生成 GB
  package identity；本任务把此循环收敛为一次 bootstrap parity。

本任务不允许出现：

```text
修改一个 trust assertion
-> Formal/Build A/Build B 全量重建
-> QA 再 full rebuild
```

未来成本按 change class 分开：

```text
kernel admission semantics / adapter / domain oracle change
  -> new kernel or adapter version
  -> seconds-level negative/metamorphic tests
  -> business full parity admission
  -> independent QA full parity admission

accepted kernel unchanged 的 stage-local C/E-only trust repair
  -> seconds-level fixture/regression tests
  -> new C/E/composite identity
  -> package-only admission and exact R binding
  -> QA mandatory seconds-level identity/negative verification
  -> QA full source-semantic admission is risk-based sampling,
     not a mandatory step for every C/E-only repair
```

只有同时满足下列条件，才能使用第二条快速路径：

- R inventory/identity exact unchanged；
- accepted kernel source/API/negative matrix exact unchanged；
- domain oracle 和 stage adapter exact unchanged；
- surface data semantics、authoritative sources 和 decision-time contract
  exact unchanged；
- mutation 仅属于 runtime contract metadata 或 publication envelope；
- surface matrix 明确把 change 分类为 C/E-only；
- QA 可随时要求 full admission；一旦分类有争议，默认回到 full parity。

分层 identity 的价值因此是同时避免数据重建和不必要的全量 source replay，
但不允许把真实 kernel/domain semantic change 伪装成廉价 C/E repair。

## 16. 建议正式任务边界

建议作为一个独立 formal task 派发，业务线程类型：

```text
业务线程-python/research-infra
```

建议标题：

```text
RESEARCH-PACKAGE-TRUST-KERNEL-LAYERED-IDENTITY-AND-STAGE4-PARITY
```

前置任务：

- `0815T003` 已通过；
- v1 Stage 5 及之后未派发项已由 v2 取代；
- 本任务 QA 通过后，才允许派发 v2 Stage H0-A。

建议 files 范围：

```text
examples/hyperliquid/research_package_trust/**
examples/hyperliquid/research_package_trust_cli.py
examples/hyperliquid/research_package_trust_stage4_adapter.py
examples/hyperliquid/test_research_package_trust_*.py
baselines/research_package_trust_kernel/**
.workflow/workflow-kit/**
.workflow/contracts/<TASK_ID>-surface-matrix.json
.workflow/runners/<TASK_ID>_archive_stage4_to_amdserver.sh
.workflow/tasks/<TASK_ID>.md
.workflow/reports/<TASK_ID>-business.md
AGENTS.md
```

明确禁止修改：

```text
local_live_analysis/...stage04_jul30_episode_v3/**  # accepted formal package
99 accepted research files
Stage 1/2/3 accepted packages
Episode/detector/domain semantics
v2 Stage H0 research outputs
```

例外 filesystem action：

- exact 删除七个 frozen empty paths；
- exact 创建 amdserver durable archive；
- package 外创建 local compatibility envelope。

## 17. 执行顺序结论

本任务是 v2 Stage H0-A 之前的当前唯一正式任务边界。

正确顺序是：

```text
冻结 Stage 4 anchors
-> kernel 小 fixture
-> 98/36/12 hostile matrix
-> identity layering
-> Stage 4 full admission parity
-> workflow template enforcement
-> amdserver durable archive
-> empty-dir cleanup
-> independent QA
-> controller 决定是否派发 v2 Stage H0-A
```

在独立 QA 接受 kernel 前：

- v2 Stage H0-A/H0-B 不派发；
- future stage 不得自称已 pin accepted kernel；
- `0815T003` accepted formal package 仍是唯一接受的 Stage 4 package；
- candidate kernel、schema 和 parity evidence 只能称为待验收 trust
  infrastructure，不能替代 Stage 4 研究事实源。

## 18. Review Remediation Closure

`2026-08-20` review 逐项处置：

| Review item | Resolution |
| --- | --- |
| Issue 1 surface schema | 冻结完整 JSON Schema、SHA、unavailable 词表和 cross-object validator rules |
| Issue 2 version pin | 冻结 registry path/schema、空 bootstrap、task/runtime pin syntax 和总控 promotion ownership |
| Issue 3 amdserver/archive | 只读确认 `molly`、parent/write/free-space/runtime，冻结 worktree 外 final path 和 false portability receipt |
| Issue 4 cleanup | 增加两阶段全体 precheck/recheck、exact `os.rmdir`、零递归/模糊删除 |
| Issue 5 hostile-first | 增加 content-addressed preflight receipt 和 full-admission runner gate |
| Issue 6 cost | 区分 bootstrap 双 full admission 与 future C/E-only risk-based QA replay |
| Issue 7 reverse binding | 增加 R mutation 使用旧 C/E 必拒绝、valid reseal 必须 C/E/composite 全更新 |
| Symlink policy | Kernel v1 绝对禁止全部 symlink，无内部相对例外 |
| Task classification | 只认显式 `task_type` + `produces_research_package`，禁止路径推断 |
| Remote process closure | transfer 前台 `wait` 完成，receipt/PID ledger 证明无后台残留 |
| Acceptance provenance | execution plan、QA、parity、inventory 一起归档进 `v1_acceptance_package/`，并为该 baseline 增加精确 `.gitignore` 例外 |

该 closure 只表示方案已具备可派发判据，不表示 kernel v1 已实现或接受。
