```md
执行线程：
- 测试线程

任务ID：
- 0512T008

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 正常验收。重点检查 T008 是否完成 `hbt.depth` live/replay view 确认、audit/schema coverage、overlay 语义确认、Stage 6J no-overlay view comparison、raw Binance depth sequence quality，并且没有改策略代码、没有运行新 replay、没有启动 live。

files：
- .workflow/tasks/0512T008.md
- .workflow/runners/run_0512T008.py
- .workflow/reports/0512T008-business.md
- local_live_analysis/t008_market_data_view_quality/T008_MARKET_DATA_VIEW_SUMMARY.md
- local_live_analysis/t008_market_data_view_quality/t008_sample_summary.csv
- local_live_analysis/t008_market_data_view_quality/t008_audit_schema_coverage.csv
- local_live_analysis/t008_market_data_view_quality/t008_converted_npz_schema.csv
- local_live_analysis/t008_market_data_view_quality/t008_raw_depth_sequence_quality.csv
- local_live_analysis/t008_market_data_view_quality/t008_view_field_comparison.csv
- local_live_analysis/t008_market_data_view_quality/t008_view_group_comparison.csv
- local_live_analysis/t008_market_data_view_quality/t008_view_compare_meta.csv
- local_live_analysis/t008_market_data_view_quality/t008_summary_metrics.json
- task_plan.md
- progress.md
- findings.md

action：
- 新建 `0512T008` 正式任务，写入计划与验收方案。
- 新建只读 runner `.workflow/runners/run_0512T008.py`。
- 读取四个现有样本：
  - `5-11-night-active`
  - `5-10-day-control-1h-06`
  - `5-9-noon`
  - `5-9-small`
- 比较 live audit、`market_state_overlay=audit` audit replay、T005 Stage 6J baseline no-overlay audit。
- 解析本地 raw Binance gzip，检查 depthUpdate / snapshot / bookTicker 事件、`U/u/pu` 连续性和 timestamp latency。
- 检查 converted npz schema 是否保留 Binance update ids。
- 未修改策略代码，未运行新的 replay/sweep，未启动 live。

verify：
- `python3 .workflow/runners/run_0512T008.py` -> exit 0，输出目录 `local_live_analysis/t008_market_data_view_quality/`。
- Runner 首轮发现 `5-9-noon` raw gzip 尾部 EOF，已增加只读容错并记录 `gzip_eof_error=1`；第二轮完成四样本统计。
- `python3 -m py_compile .workflow/runners/run_0512T008.py` -> exit 0。
- `python3 .workflow/build_dashboard.py` -> exit 0，Loaded 14 tasks and 27 reports。
- `git status --short examples/binance_tick_mm` -> 无输出，策略代码未修改。
- 输出文件存在：
  - `T008_MARKET_DATA_VIEW_SUMMARY.md`
  - `t008_sample_summary.csv`
  - `t008_audit_schema_coverage.csv`
  - `t008_converted_npz_schema.csv`
  - `t008_raw_depth_sequence_quality.csv`
  - `t008_view_field_comparison.csv`
  - `t008_view_group_comparison.csv`
  - `t008_view_compare_meta.csv`
  - `t008_summary_metrics.json`

done：
- 核心结论：
  - `strategy_core.decide_actions()` 不直接接收 full depth。live/backtest 外层先从 `hbt.depth(0)` 读取 best bid/ask、mid、top5、fair/reservation/half_spread、target ticks，再把压缩后的 action inputs 传入 shared action core。
  - live/backtest 都调用 `hbt.depth(0)`，但 shared API 不等于 shared view：live 来自 connector-maintained book，Stage 6J no-overlay 来自 replay-reconstructed book。
  - `market_state_overlay=audit` 只强制 live audit 的 compressed market/fair/target 字段；top5 tick/qty strings 仍来自 replay depth，不被 overlay。
  - 当前 audit 对现有简单策略的 compressed action-path alignment 基本够用，但不足以证明 full L2 / queue / OFI / microprice 等价。
  - converted npz schema 为 `ev|exch_ts|local_ts|px|qty|order_id|ival|fval`，不保留 Binance `U/u/pu` 或 `lastUpdateId`。raw gzip 有这些字段，但 backtest/audit 决策行没有暴露。

- 主样本 `5-11-night-active`：
  - live decision rows `802999`。
  - audit replay overlay mode `audit`，overlay count `802996`。
  - Stage 6J baseline overlay mode `off`，overlay count `0`。
  - audit replay overlay compressed market mismatch `0`，target tick mismatch `0`，但 top5 mismatch `11.0159%`。
  - Stage 6J no-overlay lag<=250ms：best bid mismatch `9.5306%`，best ask mismatch `9.5246%`，target bid tick mismatch `10.2907%`，target ask tick mismatch `10.0460%`。
  - raw depth events `536878`，snapshot `1`，bookTicker events `4436118`，`pu` mismatch `0`。

- sanity 样本：
  - `5-10-day-control-1h-06` Stage 6J no-overlay best bid/ask mismatch about `2.88%`，target tick mismatch about `4.5%-4.9%`。
  - `5-9-noon` Stage 6J no-overlay best bid/ask mismatch about `2.52%`，target tick mismatch about `3.1%`。
  - `5-9-small` Stage 6J no-overlay best bid/ask mismatch about `4.1%-4.2%`，target tick mismatch about `5.1%`。
  - 四个 raw gzip 的 `pu` mismatch 均为 `0`；`5-9-noon` gzip 尾部不完整但已记录，前面可解析数据完成统计。

- 业务判断：
  - T008 完成了数据层/view gate 的第一轮确认。
  - 当前样本可以继续支持现有 compressed strategy audit/action alignment。
  - 不应直接进入 microprice / OFI / queue 策略研究或实现，因为现有 audit/npz 没有 full L2/top-N、update ids、bookTicker-vs-depth 决策行一致性和 per-decision book provenance。
  - 建议下一步新建 T009：数据层/audit schema 修改设计与最小实现。T009 应补 top-N book、`U/u/pu`/`lastUpdateId`、exchange/local timestamps、bookTicker-vs-depth consistency、decision-row book provenance。

blockers：
- 无执行阻塞。
- 后续 microprice / OFI / queue 阻塞：现有 audit/npz schema 不足，需要 T009 数据层/audit schema 扩展后再做特征研究。

commit：
- 无

提交信息：
- 无
```
