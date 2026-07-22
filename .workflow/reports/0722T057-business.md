# 0722T057 Business Report

执行线程：
- 总控 / 业务执行线程

任务ID：
- 0722T057

状态：
- 待验收

更新时间：
- 2026-07-22 14:32 Asia/Shanghai

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `examples/hyperliquid/basis_positive_row_level_read_only_generator.py`
- `examples/hyperliquid/test_basis_positive_row_level_read_only_generator.py`
- `examples/hyperliquid/test_hyperliquid_tiny_live_m2_event_driven_watcher.py`
- `.workflow/reports/0722T056-business.md`
- `.workflow/tasks/0722T057.md`
- `task_plan.md`
- `progress.md`
- `findings.md`

action：
- Legacy relocation 在 lexical `relative_to` 后拒绝 `..`，并在 target
  存在后验证 resolved candidate 属于 resolved `PROJECT_ROOT`；symlink
  escape 保持原 recorded path。
- 新增 `HistoricalArtifactMissingError`，只表示 source artifact 真正缺失。
- Historical availability probe 只捕获 `FileNotFoundError` 和上述专用
  missing 异常；JSON、canonical count、sample mode/status 等 semantic
  错误继续传播并让测试失败。
- 新增 traversal、symlink escape、malformed-existing package hostile
  regressions。
- Resting-interval unit test 注入严格递增的 test-only `_now_ms`；production
  timestamp、manager、watcher 和 lifecycle 代码未改变。
- T056 business/controller tracking 中 implementation full SHA 修正为
  `d5e5318baaade3031f439b78aa9a7263ab0d85a6`。

verify：
- Focused portability + exact watcher interval：
  `16 passed, 2 skipped in 0.42s`。
- Exact watcher interval repeated：
  `20/20 passed`。
- Full Hyperliquid run 1：
  `1254 passed, 2 skipped in 57.46s`。
- Full Hyperliquid run 2：
  `1254 passed, 2 skipped in 55.95s`。
- `python -m py_compile` 通过。
- `git diff --check` 通过。

done：
- T056 两个 P1 均有 hostile regression 并修复。
- 真正缺失与已存在但损坏的 package 已严格分离。
- Full-suite resting-interval test 不再依赖真实整数毫秒边界。
- 未修改 production strategy、alpha、quote、risk、timestamp 或 lifecycle
  行为。

blockers：
- 独立 QA 验收。

commit：
- pending

提交信息：
- `Harden artifact portability gates`
