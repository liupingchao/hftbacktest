# 线程回报

执行线程：
- 业务线程-python/offline-evidence

任务ID：
- 0714T004

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0714T004.md`
- `.workflow/reports/0714T004-business.md`
- `examples/hyperliquid/cross_exchange_quote_fill_probability_evidence_0714T004.py`
- `examples/hyperliquid/test_cross_exchange_quote_fill_probability_evidence_0714T004.py`
- `local_live_analysis/cross_exchange_quote_fill_probability_evidence_0714T004/`
- `docs/cross_exchange_resting_interval_public_flow_auto_loop_plan.md`
- `task_plan.md`
- `progress.md`
- `findings.md`

action：
- Created `0714T004 / T011-OFFLINE-QUOTE-FILL-EVIDENCE-RERUN-WITH-V2-RESTING-INTERVAL-ARTIFACTS`.
- Implemented an offline-only v2 quote/fill evidence rerun using accepted source package:
  - `local_live_analysis/cross_exchange_resting_interval_v2_live_evidence_0714T003_20260714T063004Z/`
- Added a focused runner:
  - `examples/hyperliquid/cross_exchange_quote_fill_probability_evidence_0714T004.py`
- Added focused tests:
  - `examples/hyperliquid/test_cross_exchange_quote_fill_probability_evidence_0714T004.py`
- Generated official output package:
  - `local_live_analysis/cross_exchange_quote_fill_probability_evidence_0714T004/`

verify：
- `python -m py_compile examples/hyperliquid/cross_exchange_quote_fill_probability_evidence_0714T004.py` passed.
- `python examples/hyperliquid/cross_exchange_quote_fill_probability_evidence_0714T004.py --help` passed.
- `python examples/hyperliquid/cross_exchange_quote_fill_probability_evidence_0714T004.py` passed.
- `python -m pytest examples/hyperliquid/test_cross_exchange_quote_fill_probability_evidence_0714T004.py -q` passed: `3 passed`.
- Generated artifact JSON/CSV parse check passed:
  - JSON files `4`
  - CSV files `6`
- Deterministic rerun passed after normalizing output path fields in manifest:
  - matrices, boundary, input manifest, final route, validation report, and normalized quote/fill manifest matched.
- `git diff --check` passed.

done：
- Implementation commit:
  - `fb632b8 / Implement 0714T004 v2 quote fill evidence rerun`
- Source package:
  - `local_live_analysis/cross_exchange_resting_interval_v2_live_evidence_0714T003_20260714T063004Z/`
- Source task:
  - `0714T003`
- Source validation:
  - windows `3`
  - sha256 reconciliation passed
  - boundary status `pass`
  - JSON/CSV parse errors `0`
- Output package:
  - `local_live_analysis/cross_exchange_quote_fill_probability_evidence_0714T004/`
- Output files:
  - `attempt_level_quote_fill_evidence_matrix.csv`
  - `resting_interval_public_trades_depletion_summary.csv`
  - `public_stream_coverage_evidence_matrix.csv`
  - `censoring_horizon_matrix.csv`
  - `same_side_depth_proxy_matrix.csv`
  - `input_source_manifest.json`
  - `boundary_manifest.json`
  - `quote_fill_probability_manifest.json`
  - `final_route.json`
  - `validation_report.md`
  - `sha256_manifest.csv`
- Output summary:
  - attempt rows `71`
  - no-submit/skipped rows `70`
  - submitted/resting/no-fill rows `1`
  - public-trade summary rows `1`
  - public-stream coverage evidence rows `1`
  - censoring rows `71`
  - depth proxy rows `71`
  - fills `0`
  - post-only rejects `0`
- Resting attempt evidence:
  - physical window dir `window_03`
  - `order_attempt_id=1`
  - `attempt_key=window_01:attempt_1`
  - event sequence `1310`
  - order status `resting`
  - no fill observed
  - same-side visible qty at/ahead of quote `0.00673 BTC`
  - required depletion qty `0.00709 BTC`
  - queue depletion multiple `0`
  - lifecycle status `proxy_interval_from_local_order_response_and_cancel_ack`
  - depth status `l2_snapshot_proxy_not_after_order_resting`
  - public stream coverage status `coverage_not_proven_complete`
  - zero public trade interpretation `artifact_gap_not_no_exchange_trades`
- Final route:
  - `route_to_public_flow_artifact_repair`
- Route rationale:
  - The accepted v2 live window has a real submitted/resting/no-fill lifecycle, but `public_stream_coverage.csv` is not complete.
  - Zero captured interval trade rows remain an artifact gap, not evidence of low fill probability.
- Unsupported:
  - fill probability
  - quote policy design
  - queue priority
  - fee/rebate
  - realized PnL
  - maker viability
  - `T012`
  - promotion
  - final MVP pass
  - threshold / quote-envelope / order-size / max-submission changes

blockers：
- No fill occurred.
- Public stream coverage for the single resting attempt is `coverage_not_proven_complete`.
- Zero captured public-trade rows are explicitly `artifact_gap_not_no_exchange_trades`, not proof of no exchange public trades.
- Lifecycle and depth remain proxy-statused.

commit：
- fb632b8

提交信息：
- Implement 0714T004 v2 quote fill evidence rerun
