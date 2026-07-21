# 0721T033 Business Report

执行线程：
- 总控 auto-loop / 业务执行线程

任务ID：
- 0721T033

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `examples/hyperliquid/hyperliquid_tiny_live_m2_fill_window.py`
- `examples/hyperliquid/cross_exchange_t024_same_window_acceptance.py`
- `examples/hyperliquid/test_hyperliquid_tiny_live_m2_fill_attribution.py`
- `examples/hyperliquid/test_cross_exchange_t024_same_window_acceptance.py`
- `examples/hyperliquid/test_hyperliquid_tiny_live_m2_event_driven_watcher.py`

action：
- Producer audit 和 Task 12 独立 audit 现在将 `delayed_one_call_history_v1` 绑定为 exact `4.0s` propagation、`0.5s` final snapshot reserve、`5` direct rounds、`5.0s` total budget 和每 reference `1` 次 history。
- Zero、shorter、longer、missing、boolean/string type-confused timing 以及非 exact rounds/budget/call cap 均 fail closed。
- Task 12 对 T031+ 按 task rollout 无条件要求 manager-resting evidence，不能通过删除全部派生 CSV/snapshot summary 把合同降级为 absent。
- Quarantine artifact 即使零行也必须存在且使用 exact ordered header；每行必须具有 exact key set、canonical integer fields 和完整 canonical-row equality。
- 删除、坏 header、额外列/单元格、missing key、invalid integer、重复/伪造 row 或 reason 相同但其他字段不同均被 hostile tests 拒绝。
- Acceptance schema 提升为 `cross_exchange_principal_task12_same_window_acceptance_v10`。
- 旧 watcher 测试改为 advancing monotonic clock，在不修改 production `4.0s` 常量且不真实等待四秒的情况下验证 not-before。
- 两轮只读 hostile review：首轮发现 manager contract 全删除旁路和 DictReader extra-cell/invalid-int projection 问题；修复后复核无 P0/P1/P2 findings。
- Live executor timing、endpoint、quote、risk、size、submission、side-set 和所有 adaptive activation 未改变。

verify：
- Focused five-file regression：`514 passed in 27.47s`。
- Full Hyperliquid regression：`1017 passed in 47.87s`。
- Python compile、`git diff --check` 通过。
- T031 estimator replay 仍 exit `1`：persisted/rebuilt exposure `6/6` match，旧证据缺少 `2` 条 censor、persisted quarantine `1`，snapshot fail closed。
- T031 acceptance 仍 exit `2`：provenance `112/112`、config `72/72`、decision `43/43`、lifecycle/evidence `55 pass / 23 fail`。
- T026 exact acceptance 仍 exit `0`，lifecycle/evidence `78/78 pass`。
- T016/T022 exact acceptance 均 exit `2`，lifecycle/evidence 均为 `66 pass / 12 fail`。
- 全程 offline；未触发 live、private/account、order、cancel、network、remote 或 service 操作。

done：
- T032 QA 的两个 P1 以及实现期 hostile review 的 manager-contract/row-shape 旁路均已 fail closed。
- 修复 source 已准备进入独立 QA；通过前不允许新 private read、live 或 adaptive/multi-level activation。

blockers：
- 独立 QA acceptance。

commit：
- 7369e16a9b2a7c47a99e7ac6c3e157cc85d192de

提交信息：
- Repair exact history and quarantine gates
