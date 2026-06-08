# QA 验收结果

执行线程：
- QA验收线程

任务ID：
- 0609T001

状态：
- 已通过

更新时间：
- 2026-06-09 01:34 CST

验收线程：
- QA验收线程

验收对象：
- 业务线程-research / 0609T001

验收范围：
- 验收 `0609T001` 是否按任务边界完成 read-only basis-positive wrong-way decomposition and targeted sample design，并确认未采集新数据、未实现策略、未输出 case-library / shadow decision / private/order endpoint / live/default-on/tiny-live / promotion claim。

验收步骤：
1. 读取 `.workflow/tasks/0609T001.md` 和 `.workflow/reports/0609T001-business.md`。
2. 解析官方 artifact manifest。
3. 复跑 help、py_compile、focused pytest。
4. 复跑正式 runner 到 `/tmp/qa_0609T001_basis_wrong_way`。
5. 检查 `/tmp` outputs、boundary text 和 `git diff --check`。

实际结果：
- 官方和 `/tmp` 复现结果一致：final recommendation 为 `targeted_collection_ready`。
- `basis > 0`: `2425` rows，hit rate `0.94600939`，mean future move `45.67216495` ticks，wrong-way rows `69`，p95 wrong-way loss `138` ticks。
- `basis <= 0`: `7564` rows，hit rate `0.33853760`，mean future move `-16.94407721` ticks。
- Controlled support 通过：`binance_momentum_bucket=true`，`hl_book_state_bucket=true`。
- Top visible filter hypothesis 为 `basis_positive_small`，classification `promising_visible_filter`。
- Focused pytest 通过：`7 passed`。
- Boundary text check 和 `git diff --check` 通过。

验收结论：
- 已通过
- 结论说明：
  - `0609T001` 按任务边界完成只读 wrong-way decomposition 和 targeted sample design；结论只授权后续单独派发采样任务，不授权策略、private/order、case-library、shadow decision、live/default-on/tiny-live、parameter search 或 promotion。

通过项：
1. T006 prerequisite、canonical source-lock 和 required artifacts 均通过。
2. Runner 输出 required artifacts，并使用允许的 final recommendation taxonomy。
3. Focused verification 和 `/tmp` 复现通过。

不通过项：
1. 无

缺陷清单：
1. 无

阻塞项：
- 无

建议总控下一步：
1. 正式派发并执行 `0609T002`。
2. `0609T002` 必须保持 public-only remote collection on `awsserver1` + local processing/testing 边界。

提交信息：
- commit：`e4258b5`, `996c81c`
