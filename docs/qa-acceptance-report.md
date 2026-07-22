# QA 验收结果

## Findings（按严重性）

- P0：无。
- P1：T064 source snapshot 缺本地命令回执/源文件，10 hashes 与细分
  counts 无法独立重算。
- P1：validator 未绑定 expected filenames、hex/content hashes、receipt
  identity 或 CSV/JSON-derived counts。
- P2：eligibility、blocking reasons 和 recommendation 是 hard-coded。

执行线程：
- QA验收线程

任务ID：
- 0722T064

状态：
- 未通过

更新时间：
- 2026-07-22 17:06 Asia/Shanghai

验收对象：
- implementation
  `e96051850c7622b453dd1e94da07df9f16a133d5`

实际结果：
- Watcher candidate-before-evidence ordering 通过。
- Committed snapshot 内部支持 combined dynamic buy
  `4 observations / 1 distance` 和 feedback `0 eligible / 0s`。
- Source receipts/files 不在 checkout，独立重算失败。
- 伪 command/hash/variation 声明可通过 validator。
- Focused `3 passed in 0.03s`；no-live boundary 通过。

验收结论：
- 未通过。
- 结论说明：
  - 必须 materialize structured receipts/source files，并从真实内容计算
    eligibility 和 recommendation。

通过项：
1. Timeline ordering。
2. No-live safety。

不通过项：
1. Source provenance/content verification。
2. Derived facts/recommendation computation。

缺陷清单：
1. 见 `.workflow/reports/0722T064-qa.md`。

阻塞项：
- Repair QA 前不得进入 dynamic seed implementation。

建议总控下一步：
1. 派发 offline/read-only source-materialization repair。

提交信息：
- QA commit：由本报告提交后的线程回报提供。
