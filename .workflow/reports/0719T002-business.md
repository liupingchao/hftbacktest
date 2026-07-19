# 线程回报

执行线程：
- 总控 auto-loop / 业务实现线程

任务ID：
- 0719T002

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0719T002.md`
- `examples/hyperliquid/hyperliquid_tiny_live_m2_fill_window.py`
- `examples/hyperliquid/hyperliquid_tiny_live_m2_public_watcher.py`
- `examples/hyperliquid/cross_exchange_t024_same_window_acceptance.py`
- `examples/hyperliquid/test_hyperliquid_tiny_live_m2_fill_attribution.py`
- `examples/hyperliquid/test_hyperliquid_tiny_live_m2_fill_loop.py`
- `examples/hyperliquid/test_hyperliquid_tiny_live_m2_event_driven_watcher.py`
- `examples/hyperliquid/test_cross_exchange_t024_same_window_acceptance.py`

action：
- 删除 aggregate `any cancel succeeded` 判定，新增 `per_attempt_reference_cancel_reconciliation_v1`。
- 使用 `attempt + oid + cloid` 生成 deterministic reference key；每个 submitted reference 单独记录 matched cancel、authoritative success、ambiguous generic 和 failure 状态。
- 仅当每个 reference 至少有一个同 attempt、同 target 的 authoritative successful cancel，且不存在 missing/unknown/ambiguous cancel mapping 时，zero-fill 才可返回 `no_fill_reconciled`。
- 同 reference 已有 authoritative success 后的冗余 `already canceled, or filled` 可保留并容忍；跨 attempt/reference 的成功不能复用。
- standalone runner 不再覆盖旧 `tracked_refs`，而是保留每次提交的全部引用；immediate/final cancel rows 均写入 attempt、oid、cloid。
- inline watcher 的单边和 exchange-reconciled two-sided manager 路径均写入 attempt-bound reference/cancel evidence。
- acceptance 不信任 producer 汇总布尔值，独立核对 reference/evidence rows、唯一 key、target identity、authoritative evidence、row counts，以及 `cancel_shutdown_proof` 与 fill manifest 的一致性。
- 未修改 quote selection、strategy thresholds、live envelope、activation flags 或实际报价行为。
- 本任务未调用 live/private/account/order/cancel/network/remote/service。

verify：
- QA 反例 `attempt 1 success + attempt 2 ambiguous-only` 返回 `fail_closed/no_fill_unproven`。
- 两个 reference 分别通过 oid/cloid 成功撤单可返回 `pass/no_fill_reconciled`。
- unknown target、missing target、duplicate reference、cross-token ambiguous mapping 均 fail-closed。
- same-reference authoritative success 后 redundant generic cancel 保持 reconciled。
- standalone 两次提交产物保留 attempt 1/2 两个 reference，所有 cancel rows 均有 target identity。
- two-sided manager 产物包含两个 attempt-bound reference，逐引用 reconciliation pass。
- acceptance 拒绝缺失逐引用证据和仅伪造汇总状态的 fixture。
- focused regression：`123 passed`。
- full `python -m pytest examples/hyperliquid -q`：`476 passed in 32.06s`。
- modified Python `py_compile`、三个 CLI `--help`、`git diff --check` 通过。

done：
- T025 QA 指出的跨 attempt/reference cancel-proof fail-open 已在 producer、execution artifacts、acceptance 和 regression 层修复。
- 每个 submitted reference 现在都有显式、可定位、可独立验收的 terminal cancel proof 状态。
- 离线实现和回归完成，等待独立 QA。

blockers：
- 独立 QA 通过前，不得创建或启动新的 live window。
- Principal Task 12 和 Task 10 single-level two-sided lifecycle gate 尚未因本离线任务而关闭。

commit：
- `7235372bb2a2d6bde043a1fee3d4abc16bb5067d`

提交信息：
- `Bind cancel proof to every submitted reference`
