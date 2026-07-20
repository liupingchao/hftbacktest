# 业务线程执行回报

执行线程：
- 总控 auto-loop / 业务执行线程

任务ID：
- 0720T025

状态：
- 待验收

更新时间：
- 2026-07-20 18:42 CST

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `examples/hyperliquid/hyperliquid_maker_order_manager.py`
- `examples/hyperliquid/hyperliquid_tiny_live_m2_fill_window.py`
- `examples/hyperliquid/cross_exchange_t024_same_window_acceptance.py`
- `examples/hyperliquid/test_hyperliquid_maker_order_manager.py`
- `examples/hyperliquid/test_hyperliquid_tiny_live_m2_fill_attribution.py`
- `examples/hyperliquid/test_cross_exchange_t024_same_window_acceptance.py`
- `.workflow/tasks/0720T025.md`
- `.workflow/reports/0720T025-business.md`
- `task_plan.md`
- `progress.md`
- `findings.md`

action：
- Manager、producer 和独立 acceptance 在 identity 分类前要求 historical status 属于现有支持的 `open`、`filled`、cancel-confirmed 或 rejected exchange 状态。
- Historical row 的 actual identity kinds 必须与 expected identity kinds 精确相等；缺失或额外 oid/cloid 均 fail closed。
- Exact row 要求所有 expected identities 一致；foreign row 要求完整 identity coverage 且所有 identity 均与 expected reference 不同。
- 保留 shared-identity conflict 的 `conflicting` 分类；partial/extra fully-disjoint row 分类为 malformed/indeterminate。
- 新增 foreign oid-only、cloid-only、empty/whitespace/unknown status、exact invalid status、single-kind expected 与 extra identity 的 hostile tests。
- 保留 T024 per-alias redaction、strict schema、canonical ASCII uint64 和 T023 bounded history recovery 行为。

verify：
- Focused manager/fill/acceptance：`434 passed in 4.49s`。
- Full `python -m pytest -q -p no:cacheprovider examples/hyperliquid`：`938 passed in 39.13s`。
- 修改模块和测试 `py_compile`、acceptance CLI `--help`、`git diff --check`、implementation `git show --check`：通过。
- 独立 hostile reviewer 首次发现 extra-identity exact-cover P1；修复后原攻击集 `failure_count=0`，focused `115 passed / 319 deselected`，最终结论为 `无 findings`。
- T016 exact replay：预期 exit `2`；decision `43/43`、lifecycle `49 pass / 12 fail`，标题 `0720T016`。
- T022 exact replay：预期 exit `2`；decision `43/43`、lifecycle `49 pass / 12 fail`，标题 `0720T022`。
- 两次 replay boundary 均为 `offline_only=true`，network/private/order/cancel/remote/new-live 全部 false。
- 本任务未发生 live、private/account、order、cancel、network、remote 或 service 操作。

done：
- Partial foreign、extra-identity 和 invalid-status historical rows 无法再被 clean exact row 掩盖。
- Manager、producer 和 acceptance 独立执行相同 complete-foreign exact-cover 合同。
- Complete valid fully-disjoint foreign row 仍可安全忽略。
- T024 redaction/OID 与 T023 bounded recovery、SDK、title、replay 语义保持。

blockers：
- 无；独立 QA 是当前流程节点，QA 通过前 private historical read、新 bounded live、Task 8 和 adaptive/multi-level activation 继续锁定。

commit：
- `87c9639ec49717a4dcaaa3545e5165b0803aef5f`

提交信息：
- `Require complete foreign historical identities`
