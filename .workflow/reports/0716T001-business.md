# 线程回报

执行线程：
- 业务线程-python/offline-repair

任务ID：
- 0716T001

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `examples/hyperliquid/hyperliquid_tiny_live_m2_fill_window.py`
- `examples/hyperliquid/hyperliquid_tiny_live_m2_public_watcher.py`
- `examples/hyperliquid/test_hyperliquid_tiny_live_m2_fill_attribution.py`
- `.workflow/tasks/0716T001.md`
- `.workflow/reports/0716T001-business.md`
- `local_live_analysis/cross_exchange_interval_coverage_repaired_live_evidence_0715T001_20260715T132113Z/corrected_live_fill_attribution.csv`
- `local_live_analysis/cross_exchange_interval_coverage_repaired_live_evidence_0715T001_20260715T132113Z/corrected_live_fill_attribution_summary.json`
- `local_live_analysis/cross_exchange_interval_coverage_repaired_live_evidence_0715T001_20260715T132113Z/fill_attribution_repair_report.md`

action：
- 修复 0715T001 暴露的 live fill attribution false negative。
- `live_fill_rows()` 现在：
  - 优先按 tracked oid 归因。
  - 当 oid 缺失/不可用时，仅允许 symbol/side/price/size 在 intent size budget 内的 fallback 归因。
  - 不再把缺失 liquidity 字段默认当 maker；缺失时记录为 `unknown`。
  - 输出 `attribution_status`、`attribution_source`、`source_oid_present`、`source_has_liquidity_role`、`fill_time_ms`。
- live artifact 写入新增 `user_fills_pullback_audit.json`，保留 redacted `user_fills_by_time` pullback payload 以供事后复核。
- Hyperliquid ambiguous cancel response `Order was never placed, already canceled, or filled` 不再直接进入确定 no-fill；如果 reconciliation 仍无 fill，则记录 `fill_reconciliation_required_no_fill_unproven`。
- 重新生成 0715T001 corrected attribution 派生 artifact：
  - window_01: `0.005 BTC @ 65335`
  - window_02: `0.005 BTC @ 65366`
  - window_03: no external fill match

verify：
- `python3 -m py_compile examples/hyperliquid/hyperliquid_tiny_live_m2_public_watcher.py examples/hyperliquid/hyperliquid_tiny_live_m2_fill_window.py`
- `python3 -m pytest examples/hyperliquid/test_hyperliquid_tiny_live_m2_fill_attribution.py`
  - `5 passed`
- `python3 -m pytest examples/hyperliquid/test_hyperliquid_tiny_live_m2_event_driven_watcher.py`
  - `48 passed`
- 0715T001 corrected artifact JSON/CSV parse check
  - `artifact_parse_check=pass`
- `git diff --check`
  - passed

done：
- 0715T001 的 `0 fill` 结论已被修正为 artifact fill ledger false negative。
- 修复后的 future live artifact 可以保存 user fills pullback 并做 attempt-level attribution。
- 0715T001 当前可进入 QA 验收 repair 是否成立。
- 0715T001 仍不支持 maker/taker role、fee/PnL calibration、maker viability、T012、promotion 或 final MVP pass。

blockers：
- 无 repair 执行 blocker。
- 分析边界仍存在：0715T001 external export 不含 liquidity role，因此 maker fill count 仍 unsupported。

commit：
- 待提交

提交信息：
- 待提交
