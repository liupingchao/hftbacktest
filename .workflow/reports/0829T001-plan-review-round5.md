# 研究合同独立审查

任务ID：
- 0829T001

轮次：
- Round 5

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
1. Revision 5 已正确冻结 checkpoint、seconds、hours 单位，但没有定义
   selection、evaluation 和 raw sparse denominator 为零时的 fail-closed
   行为。不同实现可能产生 NaN、0、infinity 或跳过 filter，从而改变
   selection 与最终 classification。

已确认关闭：
- 180,000 个 20ms checkpoint 明确等于一小时；
- 所有 per-hour rate 只使用 hours denominator；
- additive state influence、对称 exposure、raw sparse 口径；
- 其余既有 P0-P2 合同。

结论：
- 数据执行锁继续保持。
- Revision 6 必须拒绝 zero/non-finite selection exposure，将 zero
  evaluation exposure 归类为 not estimable，并禁止 zero raw exposure
  通过 sparse gate。
