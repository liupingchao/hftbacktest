执行线程：
- 业务线程-live

任务ID：
- 0623T007

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `examples/hyperliquid/hyperliquid_tiny_live_m2_public_watcher.py`
- `examples/hyperliquid/test_hyperliquid_tiny_live_m2_event_driven_watcher.py`
- `.workflow/tasks/0623T007.md`
- `.workflow/reports/0623T007-business.md`
- `local_live_analysis/hyperliquid_tiny_live_m2_public_shadow_source_0623T007/**`
- `task_plan.md`
- `progress.md`
- `findings.md`

action：
- Created and dispatched `0623T007` as the M2 live public-source fair-mid shadow test.
- Implemented `m2_live_public_source_shadow_v1`, a no-submit public shadow path that can connect Hyperliquid public L2/trades events and Binance public bookTicker-compatible state into the accepted `0623T006` fair-mid provider and the `0623T004` edge gate.
- Added a Binance public bookTicker provider and event-driven public shadow runner while keeping all live client, credential, private/account/order, and cancel endpoints out of the path.
- Added public-source freshness, fair-mid source, edge gate, current-candidate, no-submit, and boundary artifacts.
- Added focused tests for positive shadow would-submit, missing Binance block, stale Binance block, no endpoint calls, and artifact generation.
- Generated T007 artifacts under `local_live_analysis/hyperliquid_tiny_live_m2_public_shadow_source_0623T007/`.

source contract：
- Policy version: `m2_live_public_source_shadow_v1`.
- Accepted source path: Hyperliquid public events plus Binance public market data into `m2_decision_time_public_fair_mid_provider_v1`.
- Decision path: watcher-local current candidate -> anti-drift gate -> fair-mid source builder -> fair-value edge gate -> shadow decision.
- Submit behavior: forced no-submit; would-submit is recorded only as shadow evidence.
- Endpoint boundary: no credentials, no private/account/order endpoint, no cancel endpoint, no live client initialization, no real order submission.

artifact evidence：
- `public_shadow_acceptance_manifest.json`:
  - `accepted_mock_public_shadow_path=true`
  - `positive_shadow_would_submit_count=2`
  - `any_private_or_order_endpoint_called=false`
  - `no_submit_enforced=true`
  - `real_orders_allowed=false`
  - `next_real_canary_authorized=false`
  - `live_public_source_observed=false`
- `scenario_summary.csv` contains 7 scenarios:
  - `positive_fresh_public_shadow_would_submit`: shadow would-submit `2`, fair-mid source pass `2`, edge gate pass `2`, endpoint flags all false.
  - `missing_binance_public_state_block`: fair-mid source block `2`, edge gate block `2`, endpoint flags all false.
  - `stale_binance_public_state_block`: fair-mid source block `2`, edge gate block `2`, endpoint flags all false.
  - `wrong_symbol_block`: fair-mid source block `2`, edge gate block `2`, endpoint flags all false.
  - `insufficient_edge_block`: fair-mid source pass `2`, edge gate block `2`, endpoint flags all false.
  - `anti_drift_shadow_block`: anti-drift block `1`, no fresh touch candidate reached fair-mid source.
  - `live_public_shadow_attempt`: blocked before live-public evidence; no Hyperliquid public L2 observed.
- Live public shadow attempt blocker:
  - `public_source_disconnect:_ssl.c:1011: The handshake operation timed out`
  - `no_hyperliquid_public_l2_observed`
  - `no_fresh_touch_candidate_reached_fair_mid_source`

verify：
- `python -m pytest examples/hyperliquid/test_hyperliquid_tiny_live_m2_event_driven_watcher.py -q` -> `33 passed`
- `python -m py_compile examples/hyperliquid/hyperliquid_tiny_live_m2_public_watcher.py` -> passed
- `python examples/hyperliquid/hyperliquid_tiny_live_m2_public_watcher.py --help` -> passed
- `python examples/hyperliquid/hyperliquid_tiny_live_m2_public_watcher.py --generate-public-shadow-source-artifacts --output-dir local_live_analysis/hyperliquid_tiny_live_m2_public_shadow_source_0623T007` -> passed with local mock/public-source-compatible artifacts and a blocked live-public attempt recorded.
- `python -m json.tool local_live_analysis/hyperliquid_tiny_live_m2_public_shadow_source_0623T007/public_shadow_acceptance_manifest.json` -> passed
- `find local_live_analysis/hyperliquid_tiny_live_m2_public_shadow_source_0623T007 -type f -empty` -> no output / no empty files
- `wc -l local_live_analysis/hyperliquid_tiny_live_m2_public_shadow_source_0623T007/scenario_summary.csv local_live_analysis/hyperliquid_tiny_live_m2_public_shadow_source_0623T007/positive_fresh_public_shadow_would_submit/public_shadow_decision_matrix.csv local_live_analysis/hyperliquid_tiny_live_m2_public_shadow_source_0623T007/positive_fresh_public_shadow_would_submit/fair_mid_source_matrix.csv local_live_analysis/hyperliquid_tiny_live_m2_public_shadow_source_0623T007/positive_fresh_public_shadow_would_submit/edge_gate_matrix.csv` -> `8`, `5`, `3`, `3` lines
- `git diff --check` -> passed before implementation commit
- `git diff --cached --check` -> passed before implementation commit

done：
- T007 completed the public shadow source path and local/mock public-source-compatible acceptance artifacts.
- The positive shadow path can reach would-submit decisions through T006 fair-mid source and T004 edge gate without touching private/order endpoints.
- Missing/stale/wrong Binance public source and insufficient edge fail closed.
- Real live-public source evidence was not collected in this environment because the public stream attempt timed out during SSL handshake and observed no Hyperliquid L2.
- Next tiny real maker canary is not authorized by this task.
- T007 does not authorize true order placement, credential reads, private/account/order endpoint use, cancel endpoint use, remote refresh, final gate, T008 live ledger claim, quote-distance changes, one-tick-back, inside-spread, cap relaxation, taker/crossing behavior, default-on behavior, M3 readiness, stable PnL, or promotion.

blockers：
- No task-scoped local/mock implementation blocker.
- Live-public observation blocker: public connection attempt failed with `_ssl.c:1011: The handshake operation timed out`, so no real live-public source pass evidence exists from this environment.
- M2 remains blocked until a separately authorized live maker fill / fee / inventory / realized PnL proof exists.

commit：
- 92ec133

提交信息：
- 0623 add public shadow source
