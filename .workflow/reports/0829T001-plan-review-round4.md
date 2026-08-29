# 研究合同独立审查

任务ID：
- 0829T001

轮次：
- Round 4

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
1. `H_raw` 定义为 checkpoint count，但 `raw_cluster_rate` 被直接与
   `5 per hour` 比较。20ms cadence 下会造成 180,000 倍单位偏差，并
   可能让频繁触发 detector 错误通过 sparse-firing gate。

已确认关闭：
- novelty interval 的 additive `W_state=2000ms` influence；
- observed/null 对称 `E/H`；
- censor 前 raw sparse behavior；
- FDP、precision confidence bound 和“误检上界”错误措辞；
- selection/evaluation bank、duration index、segment-safe cluster、
  inherited callable binding、tri-state abstention 和 no-future boundary。

结论：
- 数据执行锁继续保持。
- Revision 5 必须显式冻结 checkpoint、seconds、hours 三种 exposure
  单位，并加入 180,000 checkpoint 等于一小时的 hostile test。
