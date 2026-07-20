# 0720T032 Business Report

执行线程：
- 总控 auto-loop / 业务执行线程

任务ID：
- 0720T032

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `examples/hyperliquid/cross_exchange_online_estimators.py`
- `examples/hyperliquid/cross_exchange_t024_same_window_acceptance.py`
- `examples/hyperliquid/hyperliquid_tiny_live_m2_fill_window.py`
- `examples/hyperliquid/hyperliquid_tiny_live_m2_public_watcher.py`
- `examples/hyperliquid/test_cross_exchange_online_estimators.py`
- `examples/hyperliquid/test_cross_exchange_t024_same_window_acceptance.py`
- `examples/hyperliquid/test_hyperliquid_tiny_live_m2_event_driven_watcher.py`
- `examples/hyperliquid/test_hyperliquid_tiny_live_m2_fill_attribution.py`

action：
- 新增 versioned `confirmed_resting_exposure_censor_v1` artifact，对 first usable interval-local book 之前的 bounded leading prefix 生成独立 left-censor row。
- Producer、estimator replay 和 Task 12 现在按同一顺序独立重建并 exact-compare exposure、censor 和 quarantine；artifact 缺失、schema/content 不一致或 quarantine 非空均 fail closed。
- Interior/no-book/invalid/future/out-of-order 等证据仍进入 quarantine，不会退化为 censor。
- Generic cancel unknown 的唯一 history fallback 改为 `delayed_one_call_history_v1`：最多五轮 direct、五秒总预算、每 reference 最多一次 history、四秒 propagation not-before，并为 final open-orders snapshot 保留预算。
- 持久化 planned/actual wait、not-before、history call timing、post-history snapshot timing；producer 和 Task 12 独立验证严格顺序及预算。
- 只读 hostile review 发现 persisted quarantine artifact 可缺失通过、history/snapshot timing 审计不够严格两个 P1；均已修复并增加回归测试。
- T031 原始 live evidence 保持 immutable，未修改 quote、risk、size、submission envelope 或任何 adaptive activation。

verify：
- Focused producer/replay/history/acceptance regression：`390 passed in 8.90s`。
- Full Hyperliquid regression：`992 passed in 49.28s`。
- Python compile、`git diff --check` 通过。
- T031 exact replay 仍 exit `1`：六条 exposure 可重建，但旧 artifact 缺少两条 censor、persisted quarantine 为一条且与重建不匹配；same-window acceptance 仍 exit `2`，未升级历史证据。
- T026 exact acceptance 仍 exit `0`，lifecycle/evidence `75 pass / 0 fail`。
- T016/T022 exact acceptance 均仍 exit `2`，lifecycle/evidence `63 pass / 12 fail`，保持原 terminal proof boundary。
- 全程 offline；未触发 live、private/account、order、cancel、network、remote 或 service 操作。

done：
- Shared leading left-censor artifact、deterministic three-way replay contract 和 delayed single-history recovery protocol 已实现并完成 hostile/full regression。
- 修复 source 已准备进入独立 QA；通过前不允许新 private read、live 或 adaptive/multi-level activation。

blockers：
- 独立 QA acceptance。

commit：
- a01593b545fa5cb50c69460e1e96805bf46d6a78

提交信息：
- Repair censor and delayed history protocols
