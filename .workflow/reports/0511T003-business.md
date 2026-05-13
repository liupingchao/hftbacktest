```md
执行线程：
- 业务线程-docs

任务ID：
- 0511T003

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 正常验收。重点检查 dashboard 是否从任务索引升级为能展示结果和决策含义的实验决策看板。

files：
- .workflow/build_dashboard.py
- .workflow/dashboard.html
- .workflow/dispatch_suggestions.md
- .workflow/reports/0511T003-business.md

action：
- 升级 `.workflow/build_dashboard.py`，新增实验决策看板展示层。
- 新增 `extract_decision_summary`、`extract_metric_chips`、`extract_next_steps`、`extract_findings`、`extract_qa_summary` 等摘要提取函数。
- 增强 `TaskView`，为每个任务提供 decision summary、metric chips、business summary、QA summary 和 next steps。
- 修改 dashboard HTML，新增“决策摘要”“关键指标”“业务结果”“QA 结论”“下一步”区域。
- 将事实指标提取限制到 business/QA report；任务文件只用于任务目标、范围和执行约束，避免把计划文本误当实验事实。
- 对没有 business report 的任务，明确显示“无业务回报；当前显示任务目标和验收要求。”
- 修正 markdown numbered list 解析，避免 QA 下一步建议被合并成一行。
- 移除 `.workflow/build_dashboard.py` 中具体任务 ID 的硬编码；前置约束改为从任意任务的 `前置任务` 字段通用读取。
- 增加任务类型边界：workflow/dashboard 展示任务不从自身验收文本中抽取交易实验指标，避免把示例指标误当成任务事实。
- 保持 dashboard 为静态 HTML，不引入数据库、前端框架或交互执行能力。

verify：
- `python3 .workflow/build_dashboard.py` -> exit 0，输出 `Loaded 5 tasks and 3 reports.`，写入 `.workflow/dashboard.html` 和 `.workflow/dispatch_suggestions.md`。
- `python3 -m py_compile .workflow/build_dashboard.py` -> exit 0。
- `rg "0511T001|0511T002|0511T003|0510T001|0510T002" .workflow/build_dashboard.py` -> exit 1，表示生成器源码中没有具体任务 ID 硬编码。
- `rg "Workflow Decision Dashboard|决策摘要|关键指标|业务结果|QA 结论|下一步|diagnostic_only_no_promotion|hard failures|cancel-requested|live micro test|0510T001|0510T002|0511T001|0511T002" .workflow/dashboard.html` -> exit 0。
- `rg "无业务回报；当前显示任务目标|总控可以将|如存在后续|0511T003|0510T001|0510T002|cancel-fill: 21 / 46|samples: 4|hard failures: 0|diagnostic_only_no_promotion" .workflow/dashboard.html` -> exit 0。

done：
- Dashboard 标题已升级为 `Workflow Decision Dashboard`。
- 每张任务卡现在展示：状态摘要、决策摘要、关键指标、范围、业务结果、QA 结论、下一步。
- `0510T001` 卡片可直接看到：QA 已通过、run `5-10-day-control-1h-06`、`diagnostic_only_no_promotion`、samples `1`、candidates `6`、hard failures `0`、cancel-fill `21 / 46`、cancel-fill rate `0.456445`、acceptance passed、live micro test 否。
- `0510T002` 卡片可直接看到：待验收、跨样本 Stage 6J、`diagnostic_only_no_promotion`、samples `4`、candidates `6`、hard failures `0`、live micro test 否。
- `0511T001` 卡片可直接看到：待执行、无业务回报、当前显示任务目标和验收要求、代码范围不改策略代码、rule default off。
- `0511T002` 卡片可直接看到：待执行、无业务回报、前置约束需等待 `0511T001`、rule default off。
- `0511T003` 卡片不会把自身验收标准中的 `0511T002` 前置约束误展示为自身执行约束。
- `.workflow/build_dashboard.py` 不再硬编码任何具体任务 ID；dashboard 中出现的任务 ID 均来自 markdown 任务/report 数据。
- 关键字段目前混合使用结构化字段和启发式提取：任务状态/QA/前置/提交代码来自结构化字段；decision、samples、candidates、hard failures、cancel-fill、live micro test 来自 business/QA report 文本启发式提取。
- 后续最好给 business report 增加显式 `metrics` 或 `decision_summary` 字段，减少启发式解析。

blockers：
- 无。

commit：
- 无

提交信息：
- 无
```
