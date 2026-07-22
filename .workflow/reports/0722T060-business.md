# 0722T060 Business Report

执行线程：
- 总控 / 业务执行线程

任务ID：
- 0722T060

状态：
- 待验收

更新时间：
- 2026-07-22 15:28 Asia/Shanghai

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `examples/hyperliquid/basis_positive_row_level_read_only_generator.py`
- `examples/hyperliquid/test_basis_positive_row_level_read_only_generator.py`
- `.workflow/tasks/0722T060.md`
- `.workflow/reports/0722T060-business.md`
- `task_plan.md`
- `progress.md`
- `findings.md`

action：
- Known legacy path 无条件使用 `Path.resolve(strict=False)` 解析 source
  path，并要求其属于 resolved legacy root。
- 保留 lexical parent traversal gate 和 current destination containment。
- 新增 existing legacy parent symlink + missing final child + safe current
  same-relative target hostile regression。
- 未修改 watcher、manager、strategy、timestamp、deadline、quote、risk 或
  lifecycle production code。

verify：
- Focused resolver/availability：
  `21 passed, 2 skipped in 0.43s`。
- Deadline hostile repeated：
  `20/20 passed`。
- Full Hyperliquid run 1：
  `1260 passed, 2 skipped in 56.87s`。
- Full Hyperliquid run 2：
  `1260 passed, 2 skipped in 57.25s`。
- Conda `py_compile` 和 `git diff --check` 通过。

done：
- Existing parent symlink 在 final child 缺失时仍会被 source-side
  containment 捕获。
- Final-component symlink、safe legacy source、current destination、
  relative/absolute/traversal 和 missing-vs-malformed contracts 保持通过。

blockers：
- 独立 QA 验收。

commit：
- `f73c1480e1ad47a41533a15db5d6e4db706b87cd`

提交信息：
- `Make legacy source containment unconditional`
