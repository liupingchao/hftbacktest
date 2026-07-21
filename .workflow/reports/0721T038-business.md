# 0721T038 Business Report

执行线程：
- 总控 auto-loop / 业务执行线程

任务ID：
- 0721T038

状态：
- 待验收

更新时间：
- 2026-07-21 18:14:28 CST

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `examples/hyperliquid/hyperliquid_tiny_live_m2_public_watcher.py`
- `examples/hyperliquid/cross_exchange_t024_same_window_acceptance.py`
- `examples/hyperliquid/test_hyperliquid_tiny_live_m2_event_driven_watcher.py`
- `examples/hyperliquid/test_cross_exchange_t024_same_window_acceptance.py`

action：
- 新增 canonical submit-authorization outcome，固定 precedence 为 `immediate fail_closed > anti-drift block > edge block > pass`。
- Trigger、attempt 和 submit-decision 复用同一个 primary status/reason；immediate、anti-drift 和 edge 原始 stage evidence 均独立保留。
- T038+ independent acceptance 从 raw quantitative immediate/anti evidence 重建 stage status/reason，未知 guard source、同步标签改写和 primary reason 伪造均 fail closed。
- Public-state stale 现在 exact 绑定同 event/attempt 的 raw freshness row、`post_open_orders_l2_resync_guard` source、block reason、无 post-open anti/edge，并校验 attempt freshness projection。
- Late persistent kill-switch halt 允许保留已评估 edge，但只在同 event/attempt post-open anti 唯一且为 pass 时成立；`anti block + edge + late halt` 等 producer 不可达组合被拒绝。
- Manager hold observer 使用 `manager_hold_pump_shutdown_v2`，持久化 stop request、source close、thread exit、bounded join acknowledgement、observer return 和 cancel batch 的 monotonic 时间链。
- T038+ acceptance 要求 strict booleans、bounded wait、after-wait no-inflight、适用 source close、无 close error，并将 hold 时间链 exact 绑定所有 cancel action rows。
- 新规则全部使用 T038 rollout；T037 及更早 evidence 不追溯升级。

verify：
- Canonical helper覆盖 immediate-only、anti-only、edge-only、all-pass 和 simultaneous failure。
- End-to-end dual failure选择 immediate primary，同时保留 anti block raw evidence且不调用 order endpoint。
- Hostile tests覆盖同步 stage/primary 篡改、unknown source、freshness pass forgery、stale source bypass、cross-phase anti、stale+edge、attempt freshness drift、late-halt edge pass/block及 `anti block + edge + late halt` 不可达组合。
- Hold tests覆盖 closable、bounded blocking、nonclosable、timeout、wrong type、false acknowledgement、after-wait inflight、source close failure、timeline forgery和真实 manager-cycle cancel ordering。
- Focused watcher：`109 passed in 20.71s`。
- Focused acceptance：`231 passed in 12.11s`。
- Combined four-file regression：`592 passed in 42.63s`。
- Full Hyperliquid regression：`1095 passed in 58.60s`。
- Python compile、`git diff --check` 通过。
- T037 exact replay：blocked，decision `39 pass / 4 fail`，lifecycle `78/78 pass`。
- T031 exact replay：blocked，decision `43/43 pass`，lifecycle `55 pass / 23 fail`。
- T026 exact replay：passed，decision `43/43 pass`，lifecycle `78/78 pass`。
- T016/T022 exact replay：blocked，decision均 `43/43 pass`，lifecycle均 `66 pass / 12 fail`。
- 全程 offline；未触发 live、private/account、order、cancel、network、remote 或 service 操作。

done：
- T037 的 simultaneous guard primary-cause 缺陷已在 producer 和 independent acceptance 两侧闭合。
- T038+ manager hold pass 具备可独立复验的 bounded pump shutdown 与 cancel-before/after 时间链证明。
- Stale、late halt 和 subordinate stage evidence 按真实 producer 可达阶段图验收，不再依赖可篡改标签或不可能组合。
- 历史 evidence 和 acceptance 边界保持 immutable。
- Source 已准备进入独立 QA；QA 通过前不允许新 private read、live 或 adaptive/multi-level activation。

blockers：
- 独立 QA acceptance。

commit：
- `c436ce00cee50a03c3accadc01517691880a9927`

提交信息：
- `Unify guard and hold shutdown evidence`
