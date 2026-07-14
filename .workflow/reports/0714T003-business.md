# 线程回报

执行线程：
- 业务线程-python/live-awsserver1

任务ID：
- 0714T003

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0714T003.md`
- `.workflow/reports/0714T003-business.md`
- `docs/cross_exchange_resting_interval_public_flow_auto_loop_plan.md`
- `task_plan.md`
- `progress.md`
- `findings.md`
- `local_live_analysis/cross_exchange_resting_interval_v2_live_evidence_0714T003_20260714T063004Z/`

action：
- Created and committed the formal `0714T003 / T011-CONTROLLED-SAME-ENVELOPE-LIVE-EVIDENCE-WITH-V2-RESTING-INTERVAL-CAPTURE` dispatch.
- Synced `awsserver1:/home/admin/hftbacktest-cross-exchange` to dispatch commit `678c5b9` using a local git bundle because that remote checkout still uses an old bundle `origin`.
- Ran three sequential controlled live windows on `awsserver1` under the authorized same conservative envelope:
  - `--event-driven-edge-gate-live`
  - `--hyperliquid-l2book-fast`
  - max order size `0.005 BTC`
  - quote hold `3s`
  - wait `10s`
  - max real submissions `2`
  - post-only `Alo`
- Stopped after Window 3 because the task reached the max authorized window count and captured one `submitted_resting_no_fill` lifecycle with v2 resting-interval artifacts.
- Ran independent read-only final `open_orders()` proof for all three windows.
- Pulled the complete remote artifact package back locally with `scp`.
- Generated local validation artifacts:
  - `0714T003_local_validation_summary.json`
  - `window_classification_summary.csv`
  - `local_sha256_manifest.csv`
  - `sha256_reconciliation.csv`
  - `boundary_manifest.json`
  - `validation_report.md`

verify：
- Local preflight before live:
  - `python -m py_compile examples/hyperliquid/hyperliquid_tiny_live_m2_public_watcher.py` passed.
  - `python examples/hyperliquid/hyperliquid_tiny_live_m2_public_watcher.py --help` passed.
  - `git diff --check HEAD` passed.
- Remote preflight:
  - `/home/admin/.venvs/hyperliquid-sdk-0618T002/bin/python -m py_compile examples/hyperliquid/hyperliquid_tiny_live_m2_public_watcher.py` passed.
  - Remote watcher `--help` passed.
  - Remote `git diff --check HEAD` passed.
- Remote live execution:
  - Window 1 runner exit code `0`.
  - Window 2 runner exit status became unavailable after transient SSH timeout, but the remote watcher process completed and artifacts were present; independent proof was written after SSH recovered.
  - Window 3 runner exit code `0`.
- Safety/final state:
  - Window 1 independent final open-orders proof: `0`.
  - Window 2 independent final open-orders proof: `0`.
  - Window 3 independent final open-orders proof: `0`.
  - Remote post-run watcher process check found no remaining `0714T003` watcher process.
- Pullback/local validation:
  - Remote source root: `/home/admin/hftbacktest-cross-exchange-artifacts/cross_exchange_resting_interval_v2_live_evidence_0714T003_20260714T063004Z`
  - Local destination root: `local_live_analysis/cross_exchange_resting_interval_v2_live_evidence_0714T003_20260714T063004Z`
  - Pullback method: `scp`
  - Local file count after validation: `229`
  - JSON files `93`, CSV files `114`
  - JSON parse errors `0`
  - CSV parse errors `0`
  - Benign empty files: `window_01/runner_stderr.log`, `window_02/runner_stderr.log`, `window_03/runner_stderr.log`
  - SHA256 reconciliation passed: `222` matches, `1` skipped self-manifest row.
  - True secret-write flags `0`.
  - Boundary manifest status `pass`.
- Local final check:
  - `python -m pytest examples/hyperliquid/test_hyperliquid_tiny_live_m2_event_driven_watcher.py -q` passed (`45 passed`).
  - `python -m py_compile examples/hyperliquid/hyperliquid_tiny_live_m2_public_watcher.py` passed.
  - pulled-back artifact route check passed with `route_to_0714T004_offline_quote_fill_evidence_rerun`.

done：
- Dispatch commit: `678c5b9 / Dispatch 0714T003 v2 live evidence`
- Remote execution:
  - host `awsserver1`
  - repo `/home/admin/hftbacktest-cross-exchange`
  - commit `678c5b9ebe7bd78939c7b07af8be82ce7c10dbb8`
- Remote artifact root:
  - `/home/admin/hftbacktest-cross-exchange-artifacts/cross_exchange_resting_interval_v2_live_evidence_0714T003_20260714T063004Z`
- Local artifact root:
  - `local_live_analysis/cross_exchange_resting_interval_v2_live_evidence_0714T003_20260714T063004Z`
- Window 1 classification:
  - `no_submit_fail_closed`
  - trigger found `true`
  - quote attempt rows `33`
  - live submissions `0`
  - final open-orders `0`
  - last skip reason `edge_below_required_buffer`
- Window 2 classification:
  - `no_submit_fail_closed`
  - trigger found `true`
  - quote attempt rows `32`
  - live submissions `0`
  - final open-orders `0`
  - last skip reason `outside_quality_a_b_queue_bands;missing_intent_limit_px;missing_or_nonpositive_intent_size;missing_quality_bucket`
- Window 3 classification:
  - `submitted_resting_no_fill`
  - trigger found `true`
  - quote attempt rows `6`
  - live submissions `1`
  - submitted order `buy 0.00036 BTC @ 62650.0`
  - notional `22.554 USDC`
  - post-only `Alo`
  - order status `resting`
  - post-only rejects `0`
  - fills/maker fills `0/0`
  - cancel endpoint called `true`
  - shutdown proof `pass`
  - runner final open-orders `0`
  - independent final open-orders `0`
- Window 3 v2 resting-interval artifact evidence:
  - schema `cross_exchange_resting_interval_public_flow_capture_v2`
  - contract `cross_exchange_resting_interval_public_flow_capture_contract_v2`
  - `resting_interval_lifecycle_matrix.csv` rows `1`
  - `resting_interval_public_trades.csv` rows `0`
  - `resting_start_l2_book_snapshot_at_or_after_order_resting.csv` rows `1`
  - `resting_interval_depth_depletion_matrix.csv` rows `1`
  - `public_stream_coverage.csv` rows `1`
  - zero-row interpretation `artifact_gap_not_no_exchange_trades`
  - coverage status `coverage_not_proven_complete`
  - lifecycle interval status `proxy_interval_from_local_order_response_and_cancel_ack`
  - lifecycle start/end statuses are local response/cancel proxies, not exact exchange timestamps.
- Boundary interpretation:
  - No threshold, quote-envelope, order-size, max-submission, or strategy behavior change was made.
  - No local live-submit, credential read, private/account/order/cancel endpoint call, or local market-data collection occurred.
  - Live-related collection ran on `awsserver1`; processing and validation used the pulled-back local package.
  - This task does not claim fill probability, queue priority, fee/rebate, realized PnL, profitability, stable PnL, maker viability, `T012`, promotion, final MVP pass, or parameter expansion.
- Route recommendation:
  - QA may evaluate routing to `0714T004` offline quote/fill probability evidence rerun because a submitted/resting lifecycle with v2 resting-interval artifacts was captured.
  - The evidence itself still does not support quote policy design or fill probability; `0714T004` must decide whether v2 interval evidence is sufficient or routes back to artifact repair / more conservative evidence.

blockers：
- No fills occurred; fee/rebate/realized PnL remain unsupported.
- Window 3 `public_stream_coverage.csv` is `coverage_not_proven_complete`, so zero captured interval public-trade rows still mean `artifact_gap_not_no_exchange_trades`, not no exchange public trades.
- Resting lifecycle/depth timestamps are explicitly proxy-statused where exact exchange timestamps were unavailable.
- During Window 2, SSH temporarily timed out; after SSH recovered, the remote process had completed, final open-orders proof was `0`, and artifacts were pulled back/reconciled.

commit：
- 678c5b9

提交信息：
- Dispatch 0714T003 v2 live evidence
