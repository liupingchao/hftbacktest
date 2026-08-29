# 研究合同独立审查

任务ID：
- 0829T001

轮次：
- Round 6

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
- P2：3
- P3：0

主要缺陷：
1. Zero evaluation exposure 的失败分类没有区分 30s primary 与 10s/60s
   sensitivity。
2. No-filter fold 被声明为 abstain，但后续公式仍要求存在 `f_j`。
3. Non-finite selection exposure 同时被写成 filter rejection 与 hard
   failure，优先级冲突。

结论：
- 数据执行锁继续保持。
- Revision 7 必须区分 primary/sensitivity classification，冻结
  `f_j=NONE/META_ABSTAIN` sentinel，并将负值、non-finite 或单位转换
  不一致统一归入 selection integrity failure。
