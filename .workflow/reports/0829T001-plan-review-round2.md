# 研究合同独立审查

任务ID：
- 0829T001

轮次：
- Round 2

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
- P2：3
- P3：1

主要缺陷：
1. Signal numerator 需要完整 censor exposure，但 denominator 只检查
   当前 checkpoint，可能稀释 false-fire rate。
2. `O/N/H` 没有 duration 下标。
3. Fixed cluster 没有被 segment/quality boundary 强制断开。
4. Predecessor blob binding 尚未证明新 runner 实际调用冻结 primitive。
5. `_U95` 容易被误读为置信上界。

结论：
- Round 1 的核心统计错误已闭合。
- 数据执行锁继续保持，进入 Revision 3。
