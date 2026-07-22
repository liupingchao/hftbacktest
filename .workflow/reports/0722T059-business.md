# 0722T059 Business Report

执行线程：
- 总控 / 业务执行线程

任务ID：
- 0722T059

状态：
- 待验收

更新时间：
- 2026-07-22 15:07 Asia/Shanghai

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `examples/hyperliquid/basis_positive_row_level_read_only_generator.py`
- `examples/hyperliquid/test_basis_positive_row_level_read_only_generator.py`
- `examples/hyperliquid/test_hyperliquid_tiny_live_m2_event_driven_watcher.py`
- `.workflow/tasks/0722T059.md`
- `task_plan.md`
- `progress.md`
- `findings.md`

action：
- Known legacy recorded path 如果真实存在或为 symlink，在映射前 resolve，
  并验证仍属于 resolved legacy root。
- Destination current-checkout containment 继续独立验证。
- 新增真实临时 legacy root hostile：legacy source symlink 指向外部、
  current checkout 同 relative target 合法时仍必须拒绝。
- 新增 existing safe legacy source 到 current target 的正向回归。
- Blocking-source deadline test 的 test-only scheduler tolerance 从 `0.05s`
  调整为 `0.1s`；production timeout/deadline 未改变。

verify：
- Focused hostile + interval：
  `22 passed, 2 skipped in 4.09s`。
- Full Hyperliquid run 1：
  `1259 passed, 2 skipped in 56.35s`。
- Full Hyperliquid run 2：
  `1259 passed, 2 skipped in 56.63s`。
- py_compile 和 `git diff --check` 通过。

done：
- Legacy source 和 current destination containment 均独立成立。
- Fixture portability 全部已知 QA hostile 分支均有 regression。
- 未改变 production strategy、timestamp、deadline、quote、risk 或
  lifecycle 行为。

blockers：
- 独立 QA 验收。

commit：
- pending

提交信息：
- `Validate legacy source containment`
