# 研究合同独立审查

任务ID：
- 0829T001

轮次：
- Round 3

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
- P1：1
- P2：1
- P3：1

主要缺陷：
1. Exposure mask 未覆盖 novelty interval 内每个 state 自身的 2s trailing
   influence，denominator 仍可能过宽。
2. Sparse-firing gate 没有冻结 raw detector 还是 audit-censored 口径。
3. Task 摘要仍使用“结构误检上界”措辞。

结论：
- 数据执行锁继续保持。
- Revision 4 必须冻结 additive influence window 和 raw sparse gate。
