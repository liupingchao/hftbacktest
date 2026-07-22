# 0722T058 Business Report

执行线程：
- 总控 / 业务执行线程

任务ID：
- 0722T058

状态：
- 待验收

更新时间：
- 2026-07-22 14:50 Asia/Shanghai

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `examples/hyperliquid/basis_positive_row_level_read_only_generator.py`
- `examples/hyperliquid/test_basis_positive_row_level_read_only_generator.py`
- `.workflow/tasks/0722T058.md`
- `task_plan.md`
- `progress.md`
- `findings.md`

action：
- 新增统一 `_resolve_within_project_root()`；所有 resolver 分支均先
  resolve，再验证 candidate 属于 resolved `PROJECT_ROOT`。
- Relative、current-checkout absolute、known legacy mapping 和 unknown
  absolute 不再存在绕过 containment 的 fast path。
- Legacy lexical child 继续映射到当前 repo-relative target；`..` 在映射
  前直接拒绝。
- Unknown external absolute、relative traversal、relative symlink escape、
  legacy traversal 和 legacy target symlink escape 全部 fail-closed。
- 保持 true missing skip、malformed existing fail、精确 historical
  assertions 和 T057 test-only clock 修复。

verify：
- Focused portability + exact interval：
  `19 passed, 2 skipped in 0.18s`。
- Full Hyperliquid run 1：
  `1257 passed, 2 skipped in 56.80s`。
- Full Hyperliquid run 2：
  `1257 passed, 2 skipped in 56.31s`。
- `python -m py_compile` 通过。
- `git diff --check` 通过。

done：
- 所有 recorded path 形式共享一个 resolved project-root containment
  invariant。
- T057 QA 的 existing-path 和 relative-path P1 均有回归并修复。
- 未改变 historical statistics、strategy、production timestamp、quote、
  risk 或 lifecycle 行为。

blockers：
- 独立 QA 验收。

commit：
- `16d9db34ab977efcc36db5a2676865413dce1dc6`

提交信息：
- `Enforce artifact root containment`
