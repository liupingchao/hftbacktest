# QA 验收结果

执行线程：
- QA验收线程

任务ID：
- 0625T001

状态：
- 已通过

更新时间：
- 2026-06-25 15:25 CST

验收线程：
- QA验收线程

验收对象：
- 业务线程-research 0625T001 repair

验收范围：
- 独立复验 `27c08dd` / `72f4cb4` 是否关闭上次 QA 的 effective horizon、basis/Hyperliquid venue-state conditioning 和完整提交范围 whitespace 缺陷，并保持离线 public-only/no-submit 边界。

验收步骤：
1. 复核任务、修复业务报告、修复提交和上次 QA 缺陷清单。
2. 独立运行 focused/neighboring tests、`py_compile`、CLI help，并在两个临时目录重跑正式 accepted-artifact decomposition。
3. 从原始 `pricing_signal_rows.csv` 独立计算 effective horizon，校验 conditioning/root-cause schema、正式产物一致性、安全边界和完整提交范围 `git diff --check`。

实际结果：
- focused/neighboring tests：`13 passed in 2.21s`；`py_compile` 和 CLI help 通过。
- 两次临时重跑各生成 `10` 个非空产物；归一化输出目录后逐文件完全一致。
- 正式产物与独立临时重跑逐文件一致。
- effective horizon 独立复核：
  - `100ms`: `3595` rows，effective mean `500.41724618ms`，`materially_delayed`
  - `250ms`: `3595` rows，effective mean `500.41724618ms`，`materially_delayed`
  - `500ms`: `3595` rows，effective mean `500.41724618ms`，`aligned`
  - `1000ms`: `3594` rows，effective mean `1000.55648303ms`，`aligned`
- `venue_state_conditioning.csv` 包含 `56` rows，覆盖 basis、Hyperliquid spread/top5 imbalance/microprice/join-age 五类条件。
- `1000ms` sufficiently-covered bucket mean ranges 可复核：basis `47.0199146` ticks、HL top5 imbalance `34.33546961`、HL microprice `34.23344999`、HL spread `18.65645906`、join age `4.30117147`。
- manifest 明确 `causal_claim_allowed=false`；conditioning 仅作为历史样本 diagnostic association。
- root-cause 已包含并评估 `effective_horizon_timing_mismatch`、`basis_conditioning`、`hyperliquid_venue_state_conditioning`。
- future join、missing Binance join、stale Binance source 均为 `0`；boundary flags 全为 true。
- production funnel 和 recommendation 保持不变：fresh-touch `68`，anti-drift `64/4`，fair-mid `3/1`，edge pass `0/4`，`needs_more_public_samples`。
- 修复提交未改 production watcher、live/order/connector/collector 路径；未发现网络、凭据、private/order/cancel 调用。
- `git diff --check b21afff..HEAD` 和 QA 工作树 `git diff --check` 均通过。

验收结论：
- 已通过
- 结论说明：
  - 上次 QA 的三个缺陷已全部关闭；T001 的离线 alpha/edge decomposition、timing/conditioning 分析、产物确定性和安全边界满足任务验收要求。

通过项：
1. effective horizon counts/distribution 和 wrong horizon/timing 诊断完整可复核。
2. basis 与 Hyperliquid venue-state conditioning 已实现，覆盖状态和非因果边界明确。
3. focused/neighboring regression、正式产物、独立重跑、确定性和完整提交范围 hygiene 全部通过。
4. public-only/no-submit/no-private/no-order/no-strategy-relaxation 边界保持成立。

不通过项：
1. 无

缺陷清单：
1. 无

阻塞项：
- 无任务级阻塞。Forward evidence 仍需更多 production edge rows 和 production anti-drift same-window future markout。

建议总控下一步：
1. 接受 `0625T001`，保留 `needs_more_public_samples` 结论。
2. 由总控复核 sample contract 后，按既定顺序决定是否正式派发 `0625T002`；本 QA 不自动授权 live/canary/策略放松。

提交信息：
- commit：QA 记录提交（见当前 HEAD）
