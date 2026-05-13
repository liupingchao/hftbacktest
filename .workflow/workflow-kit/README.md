# Workflow Kit

这是一套可直接复用到新项目的 Agent 工作流文档包。

适用目标：

- 延续“总控 + 业务线程 + 测试线程 + QA验收线程”的协作架构
- 统一任务派发、线程回报、QA验收、规则更新方式
- 让下一个项目不用重新发明工作流

---

## 文档清单

### 1. `workflow-manual.md`

总手册。

适合先读，内容包括：

- 固定角色
- 默认主链路
- 任务状态
- 派发规则
- 回报规则
- QA 模式
- 规则更新机制

### 2. `task-dispatch-template.md`

总控派发模板。

适合总控使用，内容包括：

- 标准派发格式
- 是否需要 QA 的写法
- 是否需要提交代码的写法
- 规则更新提醒写法

### 3. `thread-report-template.md`

线程回报模板。

适合业务线程、测试线程、QA线程统一使用，内容包括：

- 标准回报格式
- `是否进行QA验收：是/否`
- 固定 QA 话术
- commit 回报规则

### 4. `qa-acceptance-template.md`

QA 验收模板。

适合 QA 验收线程使用，内容包括：

- 正常验收模板
- 免内容验收模板
- QA 输出规则

### 5. `workflow-adoption-guide.md`

接入指南。

适合新项目初始化时使用，内容包括：

- 需要复制哪些文件
- 推荐新建哪些固定文档
- 第一个任务怎么跑起来
- 哪些规则建议保持稳定

### 6. `new-project-init-checklist.md`

新项目初始化清单。

适合在准备接入时直接照着执行，内容包括：

- 第一天必须先定下来的事
- 最小目录和文档准备
- 第一条任务怎么选
- 第一周怎么跑
- 初始化完成的判断标准

### 7. `workflow-web-field-mapping.md`

工作流网页字段映射规范。

适合做任务看板、控制台、网页抓取层时使用，内容包括：

- 网页核心对象有哪些
- 任务/回报/QA/规则更新字段
- Markdown 到结构化字段的映射
- 首页展示建议
- 排序和最小实现建议

---

## 推荐阅读顺序

如果是新项目第一次接入，建议按这个顺序读：

1. `workflow-manual.md`
2. `workflow-adoption-guide.md`
3. `task-dispatch-template.md`
4. `thread-report-template.md`
5. `qa-acceptance-template.md`
6. `new-project-init-checklist.md`
7. `workflow-web-field-mapping.md`

---

## 推荐使用方式

### 总控

先看：

- `workflow-manual.md`
- `task-dispatch-template.md`

### 业务线程 / 测试线程

先看：

- `workflow-manual.md`
- `thread-report-template.md`

### QA验收线程

先看：

- `workflow-manual.md`
- `qa-acceptance-template.md`

### 新项目初始化

先看：

- `workflow-adoption-guide.md`
- `new-project-init-checklist.md`

### 工作流网页

先看：

- `workflow-web-field-mapping.md`

---

## 最小落地建议

把这套文档包复制到新项目后，建议至少补齐以下文件：

- `AGENTS.md`
- `task_plan.md`
- `progress.md`
- `findings.md`
- `docs/thread-playbook.md`
- `docs/qa-acceptance-report.md`

然后再开始第一条真实任务。

---

## 备注

这套文档包本身不绑定任何具体项目、业务域、代码仓库或线程名称。

可以按项目实际情况修改：

- 线程名称
- 文件路径
- 模块边界
- 业务验收标准

但建议尽量保持稳定的部分是：

- 固定任务状态
- 固定任务 ID 格式
- 派发/回报使用 Markdown 代码块
- `是否进行QA验收` 字段
- QA 文档作为最终验收事实源
