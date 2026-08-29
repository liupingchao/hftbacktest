# 研究合同独立审查

任务ID：
- 0829T001

轮次：
- Round 1

状态：
- 未通过

日期：
- 2026-08-29

审查方式：
- 独立只读合同审查
- 未运行 29-cache audit
- 未读取新实验结果或 future-price outcome

缺陷分级：
- P0：0
- P1：5
- P2：2
- P3：0

主要缺陷：
1. `q95(null count)/observed count` 不能称为 FDP 上置信界或
   precision 下界。
2. Selection 与 evaluation 重用同一 null bank。
3. Retrospective comparison mask 被错误放入 detector abstention；
   decision-supported 分母方向口径未冻结。
4. 单 depth component missingness 规则不符合 precision-first。
5. Filter 后重新形成 dependence cluster 会破坏单调性。
6. 三种 null duration 分别选择 filter，不能解释为同一 detector 的
   sensitivity。
7. Source/schema/determinism/slice fail-closed 合同不够具体。

结论：
- 数据执行锁保持。
- 必须完成 Revision 2 并再次独立审查。
