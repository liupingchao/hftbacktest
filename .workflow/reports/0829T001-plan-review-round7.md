# 研究合同独立审查

任务ID：
- 0829T001

轮次：
- Round 7

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
- P1：0
- P2：1
- P3：0

主要缺陷：
1. 全局规则已将 negative/non-finite raw exposure 归入 A-1-4 integrity
   failure，但 A-1-7 仍把 `H_raw_hours <= 0 or non-finite` 写为
   not estimable，造成双重 classification。

结论：
- 数据执行锁继续保持。
- Revision 8 必须只将 `H_raw_hours == 0` 解释为 raw not estimable；
  负值、NaN、infinity 或 non-finite raw rate 必须提前归入 A-1-4，
  且不再执行 A-1-7。
