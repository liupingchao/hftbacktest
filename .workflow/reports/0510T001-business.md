```md
执行线程：
- 测试线程

任务ID：
- 0510T001

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- .workflow/tasks/0510T001.md
- local_live_analysis/5-10-day-control-1h-06/*
- local_live_analysis/archive/5-10-day-control-1h-06.tar.gz
- local_live_analysis/archive/5-10-day-control-1h-06.tar.gz.sha256
- local_live_analysis/stage6j_regime_control_cancel_fill_risk/5-10-day-control-1h-06/*
- examples/binance_tick_mm/align_live_run.py
- examples/binance_tick_mm/maker_acceptance.py
- examples/binance_tick_mm/analyze_cancel_fill_risk.py
- examples/binance_tick_mm/stage6j_replay.py

action：
- 定位已采集并拉回本地的 live 样本 `5-10-day-control-1h-06`。
- 确认 run 目录、archive、raw market data、live audit、connector/bot/collector 日志、normal replay、audit replay、alignment report、maker acceptance、cancel-fill risk 输出均存在。
- 读取 `live_alignment_summary.md`、`maker_acceptance.json`、`alignment_report_audit_replay.json`、`backtest_audit_replay_result.json`、`cancel_fill_summary.json`。
- 复跑关键脚本 `--help`，确认流程入口可调用。
- 复跑 `maker_acceptance.py` 到 `/tmp/0510T001-maker_acceptance.json`，验证 acceptance hard gates 可重现通过。
- 复跑 `analyze_cancel_fill_risk.py` 到 `/tmp/0510T001-cancel-fill-risk`，验证 cancel-fill 风险诊断可重现。
- 复跑 `stage6j_replay.py` 单样本到 `/tmp/0510T001-stage6j-replay`，验证离线规则 replay 流程可执行。
- 未启动真实 live，未连接交易所，未修改业务代码。

verify：
- `find local_live_analysis -maxdepth 3 -iname '*5-10-day-control-1h-06*' -print` -> 找到 run 目录、archive tarball、sha256、Stage 6J 风险诊断目录。
- `python3 examples/binance_tick_mm/align_live_run.py --help` -> exit 0。
- `python3 examples/binance_tick_mm/maker_acceptance.py --help` -> exit 0。
- `python3 examples/binance_tick_mm/analyze_cancel_fill_risk.py --help` -> exit 0。
- `python3 examples/binance_tick_mm/stage6j_replay.py --help` -> exit 0。
- `python3 examples/binance_tick_mm/maker_acceptance.py --alignment-report local_live_analysis/5-10-day-control-1h-06/alignment_report_audit_replay.json --backtest-result local_live_analysis/5-10-day-control-1h-06/backtest_audit_replay_result.json --out /tmp/0510T001-maker_acceptance.json` -> exit 0，`passed=true`，hard failures 为空。
- `python3 examples/binance_tick_mm/analyze_cancel_fill_risk.py --run-id 5-10-day-control-1h-06 --local-root local_live_analysis --out-dir /tmp/0510T001-cancel-fill-risk` -> exit 0，输出 `decision=collect_more_current_format_data`。
- `python3 examples/binance_tick_mm/stage6j_replay.py --local-root local_live_analysis --run-id 5-10-day-control-1h-06 --out-dir /tmp/0510T001-stage6j-replay` -> exit 0，输出 `decision=diagnostic_only_no_promotion`，`samples=1`，`candidates=6`，`hard_failures=0`。

done：
- 阶段 1 采集 live 样本：已完成。run id 为 `5-10-day-control-1h-06`，live audit 文件存在，大小约 60MB。
- 阶段 2 拉回并归档：已完成。run 目录为 `local_live_analysis/5-10-day-control-1h-06`；archive 为 `local_live_analysis/archive/5-10-day-control-1h-06.tar.gz`；archive sha256 为 `dcfdc5c144413c00022dfaeb948c23cbfc0b45e455bc7867795ce3f7468630ec`。
- 阶段 3 replay/acceptance 验收：已完成并通过。`maker_acceptance.json` 中 `passed=true`、`hard_failures=[]`；common rows 为 `48894`；action/planned/reject/throttle match 全部为 `1.0`；working semantic/blocking mismatch 为 `0/0`；API/throttle mismatch rows 为 `0`；strict replay lag gate passed，breach/drop/fail 为 `0/0/0`。
- 阶段 4 风险诊断：已完成。live rows `65889`，decision rows `48894`，fills `46`，cancel-requested fills `21`，cancel-requested fill notional rate `0.45644522426165435`。
- 阶段 5 问题类型判断：基础 replay/working/API 对齐通过，不应先修框架；风险更偏 adverse-selection / inventory-reducing cancel race。`guard_candidate_adverse_selection_count=14`，`guard_candidate_add_side_count=7`，`same_side_readd_then_cancel_fill_count=1`。
- 阶段 6 离线规则 replay：已对 `5-10-day-control-1h-06` 单样本跑通 Stage 6J。6 个候选均无 hard failure，overlays 为 `off/off/off`，lag gate 为 `True`。单样本决策为 `diagnostic_only_no_promotion`。
- 阶段 7 跨样本验证：本轮未完成，因为 0510T001 只验证当前 run 的完整闭环。已有脚本提示当前需要继续收集/纳入更多 current-format samples。
- 阶段 8 live micro test 决策：不建议进入 live micro test。原因是当前只有单样本 Stage 6J replay，决策明确为 diagnostic-only/no-promotion。
- 正确下一步建议：创建 `0510T002`，把 `5-10-day-control-1h-06` 纳入跨样本 Stage 6J 验证矩阵，至少与 daytime + night-active 样本一起比较 PnL、max position、drop rate、churn、cancel-fill source-path，再决定是否设计 adverse-selection timing rule。

blockers：
- 无。本任务目标已完成。
- 注意：策略晋级被单样本证据不足限制，这不是本任务阻塞，而是后续任务的验收约束。

commit：
- 无

提交信息：
- 无
```
