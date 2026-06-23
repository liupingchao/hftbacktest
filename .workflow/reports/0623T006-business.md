执行线程：
- 业务线程-live

任务ID：
- 0623T006

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `examples/hyperliquid/hyperliquid_tiny_live_m2_public_watcher.py`
- `examples/hyperliquid/test_hyperliquid_tiny_live_m2_event_driven_watcher.py`
- `.workflow/tasks/0623T006.md`
- `.workflow/reports/0623T006-business.md`
- `local_live_analysis/hyperliquid_tiny_live_m2_fair_mid_source_0623T006/**`
- `task_plan.md`
- `progress.md`
- `findings.md`

action：
- Implemented / accepted `m2_decision_time_public_fair_mid_provider_v1` as the task-scoped live-compatible decision-time fair-mid provider for the `0623T004` watcher-local edge gate.
- Added `DecisionTimePublicFairMidProvider` and `build_decision_time_public_fair_mid_signal`.
- Provider contract:
  - `target_symbol=BTC`
  - `horizon_ms=1000`
  - `signal_ts_ms`
  - `fair_mid_px`
  - `source`
  - diagnostics: `hl_mid_px`, `binance_mid_px`, `basis_mid_ticks`, `lead_move_ticks`, `source_age_ms`, public-state sequence fields, and `source_status`.
- Accepted source formula:
  - `fair_mid_px = current_hyperliquid_mid + conservative_binance_lead_move_ticks * tick_size`
- Accepted inputs:
  - current in-process Hyperliquid public L2/BBO
  - decision-time Binance public state with symbol, timestamp, bid/ask or mid, and conservative `lead_move_ticks`
- Forbidden live-source inputs remain:
  - offline `pricing_signal_rows.csv`
  - optimistic proxy output
  - future markout
  - realized PnL
  - private/account/order endpoint state
- Integrated provider into `run_event_driven_inline_reprice_live` behind explicit injection via `binance_public_state_provider`.
- Kept CLI `--event-driven-edge-gate-live` default fail-closed without an injected provider; no automatic live/public network source was added.
- Added `fair_mid_source_matrix.csv` and fair-mid source counters to manifests.
- Added `--generate-fair-mid-source-artifacts` for reproducible local mock/public-source-compatible T006 artifacts.
- Added focused tests for public fair-mid provider pass, missing Binance state, stale Binance state, wrong symbol, insufficient edge, provider exception, missing Hyperliquid state, and wrong horizon.

artifact evidence：
- Generated local artifacts under `local_live_analysis/hyperliquid_tiny_live_m2_fair_mid_source_0623T006/`.
- `scenario_summary.csv` contains 8 watcher-path scenarios:
  - `positive_fresh_fair_mid_pass`: fair-mid source pass `1`, edge gate pass `1`, mock `Alo` order calls `1`.
  - `missing_source_block`: edge gate block `1`, mock order calls `0`.
  - `missing_binance_public_state_block`: fair-mid source block `1`, edge gate block `1`, mock order calls `0`.
  - `stale_source_block`: fair-mid source block `1`, edge gate block `1`, mock order calls `0`.
  - `wrong_symbol_block`: fair-mid source block `1`, edge gate block `1`, mock order calls `0`.
  - `wrong_horizon_block`: edge gate block `1`, mock order calls `0`.
  - `insufficient_edge_block`: fair-mid source pass `1`, edge gate block `1`, mock order calls `0`.
  - `provider_exception_block`: fair-mid source block `1`, edge gate block `1`, mock order calls `0`.
- `provider_contract_matrix.csv` contains 5 contract/evaluator cases:
  - missing Hyperliquid public state
  - wrong horizon
  - future timestamp
  - missing fair mid
  - invalid quote/tick
- Artifact validation found 466 files and 0 empty files in the T006 artifact directory.

verify：
- `python -m pytest examples/hyperliquid/test_hyperliquid_tiny_live_m2_event_driven_watcher.py -q` -> `29 passed`
- `python -m py_compile examples/hyperliquid/hyperliquid_tiny_live_m2_public_watcher.py` -> passed
- `python examples/hyperliquid/hyperliquid_tiny_live_m2_public_watcher.py --help` -> passed
- `python examples/hyperliquid/hyperliquid_tiny_live_m2_public_watcher.py --generate-fair-mid-source-artifacts --output-dir local_live_analysis/hyperliquid_tiny_live_m2_fair_mid_source_0623T006` -> passed
- `python -m json.tool local_live_analysis/hyperliquid_tiny_live_m2_fair_mid_source_0623T006/fair_mid_source_acceptance_manifest.json` -> passed
- `find local_live_analysis/hyperliquid_tiny_live_m2_fair_mid_source_0623T006 -type f -empty` -> no output / no empty files
- `wc -l local_live_analysis/hyperliquid_tiny_live_m2_fair_mid_source_0623T006/scenario_summary.csv local_live_analysis/hyperliquid_tiny_live_m2_fair_mid_source_0623T006/provider_contract_matrix.csv` -> `9` and `6` lines, respectively
- `git diff --check` -> passed

done：
- T006 completed the decision-time public fair-mid provider contract and watcher-local edge-gate integration.
- A fresh public-state provider can now be injected into the existing edge gate; missing or invalid provider state fails closed before order submission.
- Local mock/public-source-compatible artifacts cover positive-edge pass and required fail-closed cases.
- No live order window was run, no credentials were read, no private/account/order endpoint was called, no remote checkout was refreshed, no final gate was rerun, no live data collection was performed, and no real order endpoint was called by this task.
- M2 remains blocked until separately authorized live maker fill / fee / inventory / realized PnL proof exists.
- This task does not authorize live execution, quote-distance change, one-tick-back, inside-spread, cap relaxation, M3 readiness, stable PnL, default-on behavior, or promotion.

blockers：
- No task-scoped implementation blocker.
- Forward blocker: accepted provider contract still needs a separately scoped live/public source wiring and live-risk task before any real endpoint use. M2 remains blocked on no live maker fill / fee / inventory / realized PnL proof.

commit：
- a6ee959

提交信息：
- 0623 add fair mid source provider
