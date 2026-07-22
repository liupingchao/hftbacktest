# 0722T054 Business Report

执行线程：
- 总控 auto-loop / 业务执行线程

任务ID：
- 0722T054

状态：
- 待验收

更新时间：
- 2026-07-22 11:04 Asia/Shanghai

是否进行QA验收：
- 是

前置任务：
- `0722T053` 独立 QA 为 `未通过`。
- T053 implementation：
  `68416c180ca02d53de455f3971fbeb9d5cb5ac4b`
- T053 QA records：
  `598216df86b3a16c3b4f3530b482b9d59f85bf2d`

implementation：
- `4db5fb6f061106d13f5cab16269aa1bd58926507`

边界：
- 严格 offline-only；未执行 live、private/account、真实 order/cancel、
  network、remote 或 service 操作。
- Standing envelope 保持 `0.005 BTC/order`、`0.01 BTC position`、
  `1 USDC loss`、`2 submissions/window`、`1800s/window`。

action：
- Active multi-level manager 在未显式提供 `runtime_config` 时构造失败，
  不再继承 legacy `30` submission fallback。
- Live-mode manager 必须提供 `before_submit` callback；manager 在每个
  addition 的最终 endpoint 调用前执行 callback，callback failure 发生在
  generation、submission counter 和 order state mutation 之前。
- 任一 submit 经 direct query 后仍为 unresolved `unknown`，同一 batch
  立即停止；剩余 additions 标记
  `submit_batch_stopped_after_unknown`，unknown leaves 保持 exposure-bearing。
- 新进程从 authoritative open orders 恢复 owned order 时，将
  `submission_provenance_complete=false`；允许 exact hold 和 cancel，但
  cancel-confirmed 后任何 addition 均以
  `restart_submission_provenance_incomplete` fail-closed。
- Watcher `build_task7_desired_quotes`、`build_task7_order_manager`、
  `run_task7_manager_cycle` 和 normal inline manager path 接受同一
  `QuoteLadderConfigV1`。
- Default 无 config 或 multi-level default-off 时继续构造单层 quote；
  active config 经 guarded ladder 和 strict conversion 产生 matching
  multi-order manager target。
- 首次 active multi-level 与 dynamic spread/fill feedback 保持互斥；
  default-off ladder 不阻断既有 adaptive single-level path。
- Normal manager cycle 将 persistent halt gate 作为 per-submit callback；
  live-status/config hash 和 multi-level snapshot 绑定同一 ladder config。

verify：
- T054 hostile subset：
  `6 passed, 247 deselected in 0.13s`。
- Focused manager/ladder/event-driven watcher：
  `259 passed in 21.04s`。
- Related public watcher/shared kernel/executor：
  `93 passed in 0.23s`。
- Full Hyperliquid：
  `1250 passed in 51.67s`。
- `python -m py_compile` 对 manager、watcher 和两个 focused test 文件通过。
- `git diff --check` 对 T054 scope 通过。

hostile results：
- Omitted runtime config：active multi-level manager 在 endpoint 前抛出
  `multi_level_runtime_config_required`。
- Unknown submit：四层目标仅发生 `1` 次 fake order call；后续 `3` 个
  additions blocked；submission count `1`，unknown buy leaves 继续计入
  working exposure。
- Per-submit halt：callback 第 `2` 次触发 halt 时仅有第 `1` 个 fake order
  call，submission count 保持 `1`。
- Normal watcher active ladder：产生 buy/buy/sell/sell 四个 desired rows，
  manager capacity 为两层/side；standing cap `2` 在任何 fake order call 前
  拒绝。
- Restart recovery：恢复后 exact hold/cancel 通过；cancel-confirmed 后
  readd 被 provenance gate 拒绝，未生成或复用新 cloid。

done：
- T053 的两个独立 QA P1 均有精确 hostile regression 并修复。
- Guarded ladder 已接入 normal watcher quote/manager API path，保持
  default-off。
- Per-submit halt 和 restart cancellation-only recovery 边界已落地。
- 既有 single-level/T052 lifecycle、dynamic spread 和 fill-feedback
  regression 全部保持。

limits：
- 本任务不提供 multi-level live evidence，也不将 optional watcher API
  解释为 live authorization。
- Restart 未恢复 durable historical submission/generation state；当前明确
  为 hold/cancel-only recovery，不能宣称完整 restart readiness。
- Standing cap `2` 下 fresh two-level-per-side ladder必然 zero-call
  fail-closed；任何 multi-level live 仍需新的风险 envelope 决策。
- 不支持 stable PnL、fill rate、fee/rebate、queue priority、maker
  viability、promotion 或 final MVP 结论。

blockers：
- 独立 QA 验收。

提交信息：
- Implementation commit：
  `4db5fb6f061106d13f5cab16269aa1bd58926507`
- Workflow report commit：待提交
