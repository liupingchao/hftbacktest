# 0722T056 Business Report

执行线程：
- 总控 / 业务执行线程

任务ID：
- 0722T056

状态：
- 待验收

更新时间：
- 2026-07-22 14:12 Asia/Shanghai

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `examples/hyperliquid/basis_positive_row_level_read_only_generator.py`
- `examples/hyperliquid/test_basis_positive_row_level_read_only_generator.py`
- `examples/hyperliquid/cross_exchange_lead_lag_join.py`
- `examples/hyperliquid/test_cross_exchange_lead_lag_join.py`
- `.workflow/tasks/0722T056.md`
- `task_plan.md`
- `progress.md`
- `findings.md`

action：
- 增加受限 legacy project-root 重定位，仅接受
  `/home/molly/project/hftbacktest`，且仅在对应 repo-relative artifact
  实际存在时重定位。
- T003 input directory 和 per-sample pricing-signal row 路径统一使用该
  resolver。
- 增加历史 source/sample availability probe。
- 两个 exact historical integration tests 在外部数据包缺失时显式 skip；
  数据完整时继续执行原有 `3545` 和 `3599` exact assertions。
- 增加 self-contained hostile tests，验证已知旧路径可重定位、未知绝对
  路径不被重定位、partial sample fixture 不被误判完整。

verify：
- Focused：
  `12 passed, 2 skipped in 0.15s`。
- Full Hyperliquid：
  `1251 passed, 2 skipped in 56.06s`。
- `python -m py_compile` 对四个实现/测试文件通过。
- `git diff --check` 通过。

done：
- Clean clone / 当前无外部历史数据包环境不再出现 collection failures。
- 两个 skip 均具有精确的缺失 artifact reason。
- 历史 exact-count integration contract 在数据可用环境保持不变。
- 未改变策略、alpha、quote、risk、order lifecycle 或 live 行为。

blockers：
- 独立 QA 验收。

commit：
- `d5e5318baaade3031f439b78aa9a7263ab0d85a6`

提交信息：
- `Make historical artifact tests portable`
