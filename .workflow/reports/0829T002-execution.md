# 0829T002 执行报告

日期：
- 2026-08-29

任务：
- `FRESH_CHANNEL_CONSENSUS_MSTATE_V1_A_MINUS1`

状态：
- 待验收

执行版本：
- implementation commit: `4457f127`
- classification correction commit: `54368521`
- frozen plan SHA256:
  `9f214893d7d4ef0431efaf6641c52c84d2fe4f0b516897818b92671355343de5`

验证：
- hostile tests: `19 passed`
- canonical Build A: 完成
- fresh Build B: 完成
- preseal difference count: `0`
- pending difference count: `0`
- final difference count: `0`
- non-cache artifact count: `23`
- future/outcome access: `false`

最终分类：
- `Aminus1_mstate_integrity_failed`

最早失败 gate：
- `A-1-2`
- 唯一失败条件：`slice_invariance_mismatches = 17`

完整性结果：
- M-state partition violations: `0`
- channel action partition violations: `0`
- `NEW_INVALID` actions: `0`
- unauthorized TTL refresh: `0`
- cross-segment memory carry: `0`
- anchor contract violations: `0`
- parameter monotonicity violations: `0`
- null invariant mismatches: `0`

结构结果：
- common anchors: `3,783`
- raw selected-filter exposure: `2.3545888889h`
- raw selected-filter clusters: `1`
- raw cluster rate: `0.4247025902/h`
- maximum 5s burst: `1`

Cross-fit estimators：
- 10s: observed clusters `0`, exposure `0.7171222222h`
- 30s primary: observed clusters `0`, exposure `0.8763222222h`
- 60s: observed clusters `1`, exposure `0.4267h`
- 30s null p95 false-cluster rate: `1.1411327645/h`

主要诊断：
- 异步 channel memory 将 common-anchor support 从前序方案的 `350`
  提高到 `3,783`，说明同步完整路径支持依赖确实被降低。
- `F000` 仅保留 `61/3,783` candidates，主要取消原因为
  `consensus_lost=1,939` 和 `persistence_abstain=1,767`。
- Cross-fit 常选 `F011`，全样本仅保留 `1` 个 raw cluster。
- 17 个 slice mismatch 的 M-state support count 全部精确一致，变化只在
  anchor identity。
- 根因是 per-direction 30s refractory 的 renewal 相位依赖。人工切片
  删除早期 accepted anchor 后，在密集 natural-onset 流中，后续
  refractory 选择序列可长期错相；固定 62s guard 不能保证重新同步。
- 因此这是注册 detector 的 reset-invariance 失败，不应通过放宽阈值、
  缩短 guard 或忽略 mismatch 来修复。

科学结论：
- 新 M-state 表示解决了 primitive observability 的一部分问题，但没有
  形成可稳定复现的 alignment landmark。
- 即使暂时忽略 A-1-2，30s primary observed cluster 为 `0`，后续仍会在
  A-1-5 失败；conditional null false-fire rate 也远高于冻结上限。
- 不授权 A0、future outcome、live 或 private/order execution。

建议：
- 保留三通道 event-evidence memory，不回退同步完整路径。
- 下一版本先重新注册一个 reset-invariant anchor suppression contract，
  例如基于固定 causal epoch/bucket 的 deterministic thinning，避免
  accepted-anchor-driven refractory renewal。
- 新版本必须重新从 precision-first A-1 开始；不得在当前版本上调整
  阈值或事后删除 17 个 mismatch。
