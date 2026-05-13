# 工作流网页字段映射规范

这份文档给“工作流网页 / 看板 / 控制台”使用。

目标是把任务派发、线程回报、QA 验收结果，统一映射成网页可展示、可检索、可流转的数据结构。

这份规范不绑定任何具体项目，只定义字段和显示规则。

---

## 1. 网页里到底要展示哪些对象

网页层建议只维护 4 类核心对象：

1. `任务 Task`
2. `线程回报 Report`
3. `QA验收 Acceptance`
4. `规则更新 RuleUpdate`

不要一开始就把聊天全文、完整日志、所有原始命令都当成首页对象。

首页首先要解决的是：

- 当前有哪些任务
- 每个任务现在在哪个状态
- 哪个线程在负责
- 有没有进入 QA
- QA 最终结论是什么
- 下一步该派什么

---

## 2. 对象关系

建议按下面关系组织：

```text
Task
  ├─ belongs to one 执行线程
  ├─ has many Report
  ├─ has zero or one latest Acceptance
  └─ may have zero or many RuleUpdate references
```

补充理解：

- `Task` 是主对象
- `Report` 是任务执行过程中的事实回报
- `Acceptance` 是 QA 最终裁决
- `RuleUpdate` 是规则变更通知，不替代任务本身

---

## 3. 任务 Task 字段

这是网页最核心的卡片对象。

### 3.1 必备字段

| 字段 | 类型 | 必填 | 说明 |
|---|---|---:|---|
| `task_id` | string | 是 | 固定格式，例：`0421T006` |
| `title` | string | 是 | 任务标题，给网页列表直接显示 |
| `summary` | string | 是 | 简短描述，建议 1-2 句 |
| `thread_name` | string | 是 | 执行线程名称 |
| `status` | enum | 是 | 只允许 7 个固定状态 |
| `qa_required` | boolean | 是 | 是否需要进入 QA 节点 |
| `qa_mode` | enum | 是 | `normal / exempt_content / none` |
| `execution_order` | enum | 是 | `single / parallel` |
| `needs_commit` | boolean | 是 | 是否要求提交代码 |
| `created_at` | datetime | 是 | 任务创建时间 |
| `updated_at` | datetime | 是 | 最近更新时间 |

### 3.2 推荐字段

| 字段 | 类型 | 必填 | 说明 |
|---|---|---:|---|
| `prerequisite_task_id` | string/null | 否 | 前置任务 ID |
| `rule_update_notice` | string/null | 否 | 这轮是否带规则更新提醒 |
| `scope_files` | string[] | 否 | 范围文件或模块 |
| `action_text` | string | 否 | 这轮行动要求 |
| `verify_text` | string | 否 | 这轮验证要求 |
| `done_text` | string | 否 | 回报时必须说明什么 |
| `dispatch_markdown` | string | 否 | 原始派发 Markdown |
| `dispatch_source` | string | 否 | 来源文档/来源线程/来源消息 |

### 3.3 状态枚举

只允许这 7 个值：

- `待执行`
- `执行中`
- `待验收`
- `已通过`
- `未通过`
- `阻塞`
- `作废`

网页不要额外创造：

- `验收中`
- `测试中`
- `处理中`
- `等待反馈`

如果业务上真的存在这些过程，也应映射回固定 7 状态之一。

---

## 4. 线程回报 Report 字段

`Report` 是任务执行过程中的阶段性事实。

一个任务可以对应多条回报，但网页通常只需要突出：

- 最新一条
- 最新有效一条
- 是否已准备进入 QA

### 4.1 必备字段

| 字段 | 类型 | 必填 | 说明 |
|---|---|---:|---|
| `report_id` | string | 是 | 网页内部唯一 ID |
| `task_id` | string | 是 | 关联任务 ID |
| `thread_name` | string | 是 | 回报线程 |
| `status` | enum | 是 | 当前回报状态 |
| `qa_requested` | boolean | 是 | 是否进入 QA |
| `reported_at` | datetime | 是 | 回报时间 |

### 4.2 推荐字段

| 字段 | 类型 | 必填 | 说明 |
|---|---|---:|---|
| `qa_note` | string | 否 | `当前任务结果暂不进入QA验收...` 等说明 |
| `files_text` | string | 否 | 本轮文件范围 |
| `action_text` | string | 否 | 本轮做了什么 |
| `verify_text` | string | 否 | 本轮如何验证 |
| `done_text` | string | 否 | 本轮完成了什么 |
| `blockers_text` | string | 否 | 阻塞说明 |
| `commit_id` | string/null | 否 | commit id 或 null |
| `commit_message` | string/null | 否 | commit message 或 null |
| `report_markdown` | string | 否 | 原始回报 Markdown |

### 4.3 网页显示规则

- 默认列表显示“最新一条回报”
- 若任务存在多次回报，可展开历史
- `qa_requested=false` 时，网页要明确显示“暂不进入 QA”
- `commit_id` 为空时，不要强行显示“提交失败”，只显示“无提交”

---

## 5. QA验收 Acceptance 字段

`Acceptance` 是 QA 节点的最终裁决对象。

网页里它不是普通回报卡，而是任务的最终验收层。

### 5.1 必备字段

| 字段 | 类型 | 必填 | 说明 |
|---|---|---:|---|
| `acceptance_id` | string | 是 | 网页内部唯一 ID |
| `task_id` | string | 是 | 关联任务 ID |
| `status` | enum | 是 | 只允许 `已通过 / 未通过 / 阻塞` |
| `accepted_at` | datetime | 是 | QA 更新时间 |
| `acceptor_thread` | string | 是 | 默认是 `QA验收线程` |
| `acceptance_scope` | string | 是 | 验收范围 |
| `acceptance_result` | string | 是 | 实际结果摘要 |
| `acceptance_conclusion` | string | 是 | 验收结论说明 |

### 5.2 推荐字段

| 字段 | 类型 | 必填 | 说明 |
|---|---|---:|---|
| `acceptance_mode` | enum | 否 | `normal / exempt_content` |
| `pass_items` | string[] | 否 | 通过项 |
| `fail_items` | string[] | 否 | 不通过项 |
| `bugs` | string[] | 否 | 缺陷清单 |
| `blockers` | string[] | 否 | 阻塞项 |
| `next_actions` | string[] | 否 | 建议总控下一步 |
| `acceptance_markdown` | string | 否 | 原始 QA Markdown |

### 5.3 网页显示规则

- 一个任务默认只高亮“最新有效 QA 结果”
- `免内容验收` 要单独打标签，不能和正常通过混在一起
- `已通过` 不等于“功能绝对正确”，它只表示当前 QA 结论通过
- `阻塞` 必须在任务卡上直接可见，不能藏在展开层

---

## 6. 规则更新 RuleUpdate 字段

规则更新不是任务，但网页最好有单独入口管理。

### 6.1 必备字段

| 字段 | 类型 | 必填 | 说明 |
|---|---|---:|---|
| `rule_update_id` | string | 是 | 规则更新唯一 ID |
| `title` | string | 是 | 本次规则更新标题 |
| `summary` | string | 是 | 简短说明 |
| `updated_at` | datetime | 是 | 更新时间 |

### 6.2 推荐字段

| 字段 | 类型 | 必填 | 说明 |
|---|---|---:|---|
| `affected_threads` | string[] | 否 | 受影响线程 |
| `document_path` | string | 否 | 对应规则文档 |
| `must_notify_next_task` | boolean | 否 | 是否要求在下一条任务里提醒 |
| `raw_rule_text` | string | 否 | 原始规则文本 |

### 6.3 网页显示规则

- 规则更新不要混进普通任务列表
- 应单独放在“规则更新”或“系统通知”区
- 对每个线程只需要跟踪“是否已在下一条任务中提醒过”

---

## 7. Markdown 到结构化字段的映射规则

网页如果要自动抓取文本，建议只抓固定标题字段，不抓自由散文。

### 7.1 派发文本抓取键

建议抓这些一级字段：

- `执行线程`
- `任务ID`
- `标题`
- `简短描述`
- `状态`
- `执行顺序`
- `前置任务`
- `规则更新提醒`
- `是否需要提交代码`
- `是否进行QA验收`
- `QA参与`
- `QA验收方式`
- `QA验收线程动作`
- `files`
- `action`
- `verify`
- `done`

### 7.2 回报文本抓取键

建议抓这些一级字段：

- `执行线程`
- `任务ID`
- `状态`
- `是否进行QA验收`
- `QA说明`
- `files`
- `action`
- `verify`
- `done`
- `blockers`
- `commit`
- `提交信息`

### 7.3 QA 文本抓取键

建议抓这些一级字段：

- `执行线程`
- `任务ID`
- `状态`
- `更新时间`
- `验收线程`
- `验收对象`
- `验收方式`
- `验收范围`
- `验收步骤`
- `实际结果`
- `验收结论`
- `通过项`
- `不通过项`
- `缺陷清单`
- `阻塞项`
- `建议总控下一步`
- `提交信息`

### 7.4 抓取原则

- 只抓固定标题，不抓自然段语义猜测
- 标题名尽量不要做同义词兼容，先统一模板
- 一条文本里如果出现重复字段，以最后一次为准
- 同一 `task_id` 下，`QA` 结果优先级高于普通回报

---

## 8. 网页首页推荐展示区

如果要做首页，建议只放这几类信息：

### 8.1 顶部指标

- 当前阶段
- 打开风险数
- 待确认数
- 待派发数
- 可选：桥接状态

### 8.2 中段模块

- 项目状态
- 风险与异常
- 待确认
- 待派发

### 8.3 不建议首页主位展示的内容

- 完整 commit message 长列表
- 所有线程完整聊天记录
- Bug 全量流水
- 所有历史回报全文
- 规则更新全文

这些内容应下钻到详情页，不要占首页主位。

---

## 9. 排序规则建议

### 9.1 任务列表排序

建议优先级从高到低：

1. `阻塞`
2. `未通过`
3. `待验收`
4. `执行中`
5. `待执行`
6. `已通过`
7. `作废`

同状态下再按：

- `updated_at` 倒序

### 9.2 QA 列表排序

建议按：

1. `阻塞`
2. `未通过`
3. `已通过`

同状态下按：

- `accepted_at` 倒序

---

## 10. 最小实现建议

如果网页先做第一版，建议只实现：

1. 任务卡列表
2. 任务详情页
3. 最新线程回报展示
4. 最新 QA 结果展示
5. 规则更新提醒区
6. 基础筛选：线程、状态、是否需要 QA

先不要第一版就做：

- 全聊天流展示
- 自动复杂依赖图
- 高级统计报表
- 多维自定义视图编辑器

先让“看清现在发生了什么”成立，再加高级功能。

---

## 11. 一句话原则

网页不是聊天记录阅读器，也不是流水账仓库。

网页首先应该让总控在很短时间内看清：

- 现在有哪些任务
- 谁在做
- 做到哪一步
- 有没有进入 QA
- QA 怎么判
- 下一步该派什么
