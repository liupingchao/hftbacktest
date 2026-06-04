执行线程：
- 业务线程-research

任务ID：
- 0604T007

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0604T007.md`
- `.workflow/reports/0604T007-business.md`
- `examples/hyperliquid/canonical_horizon_regime_diagnostics.py`
- `examples/hyperliquid/test_canonical_horizon_regime_diagnostics.py`
- `local_live_analysis/canonical_horizon_regime_diagnostics_0604T007/`
- `progress.md`
- `task_plan.md`
- `findings.md`

action：
- Implemented a read-only canonical horizon / regime diagnostics runner that consumes `local_live_analysis/event_mode_canonical_pricing_signal_0604T003/` through the `0604T004` canonical loader path.
- Added a task-level guard that rejects inputs with no canonical event-mode samples or any diagnostic-only synthetic samples.
- Generated task-scoped watch-only artifacts:
  - `horizon_independence_diagnostics.csv`
  - `regime_conditioning_diagnostics.csv`
  - `regime_watch_list.csv`
  - `horizon_regime_diagnostics_manifest.json`
  - `horizon_regime_diagnostics_report.md`
- Added focused tests for horizon alias classification, minimum-support classification, concentration penalty, non-canonical input refusal, and report artifact generation.

verify：
- `python examples/hyperliquid/canonical_horizon_regime_diagnostics.py --help` passed.
- `python -m py_compile examples/hyperliquid/canonical_horizon_regime_diagnostics.py examples/hyperliquid/test_canonical_horizon_regime_diagnostics.py` passed.
- `python -m pytest examples/hyperliquid/test_canonical_horizon_regime_diagnostics.py` passed: `5 passed`.
- `python examples/hyperliquid/canonical_horizon_regime_diagnostics.py --input-dir local_live_analysis/event_mode_canonical_pricing_signal_0604T003 --output-dir local_live_analysis/canonical_horizon_regime_diagnostics_0604T007` passed: `canonical_sample_count=3`, `horizon_rows=6`, `regime_rows=18`.
- `python -m json.tool local_live_analysis/canonical_horizon_regime_diagnostics_0604T007/horizon_regime_diagnostics_manifest.json` passed.
- `git diff --check` passed.

done：
- Horizon independence findings:
  - `100ms` and `250ms` are `watch_needs_more_samples` because public `l2Book` cadence can weakly alias short nominal horizons.
  - `500ms`, `1000ms`, `5000ms`, and `10000ms` are `diagnostic_supported`; `1000ms+` remains the preferred interpretation band.
- Regime conditioning findings:
  - Regime support counts are `10 diagnostic_supported`, `6 watch_needs_more_samples`, and `2 reject_aliased_or_concentrated`.
  - Regime rows include sample count, row count, direction consistency, effect concentration, row concentration proxy, and horizon-level feature/correlation context.
  - All regime rows remain watch-only diagnostics and do not define final high-confidence regimes or maker actions.
- This is only read-only horizon / regime diagnostics. It does not authorize new collection, final regime selection, case-library construction, shadow decisions, strategy implementation, private/order endpoints, order lifecycle, live/default-on/tiny-live, parameter search, or promotion.

blockers：
- 无

commit：
- 066a6a8

提交信息：
- 0604T007 canonical horizon regime diagnostics
