执行线程：
- 业务线程-live

任务ID：
- 0622T006

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `examples/hyperliquid/hyperliquid_tiny_live_m2_public_watcher.py`
- `examples/hyperliquid/test_hyperliquid_tiny_live_m2_event_driven_watcher.py`
- `.workflow/tasks/0622T006.md`
- `.workflow/reports/0622T006-business.md`
- `task_plan.md`
- `progress.md`
- `findings.md`
- `local_live_analysis/hyperliquid_tiny_live_m2_anti_drift_gate_0622T006/**`
- `local_live_analysis/hyperliquid_tiny_live_m2_anti_drift_gate_0622T006_rerun/**`
- `local_live_analysis/hyperliquid_tiny_live_m2_anti_drift_gate_0622T006_rerun2/**`

action：
- Implemented `--event-driven-anti-drift-live` in the existing watcher-local inline reprice path.
- Added rolling public BBO / trade-flow state and an anti-drift / touch-stability gate with `bbo_lookback_ms=750`, `min_stable_ms=250`, `flow_lookback_ms=1000`, `pressure_ratio_threshold=2.0`, and `min_pressure_qty_btc=0.01`.
- Preserved Hyperliquid post-only `Alo`, unchanged dynamic size hard cap `<=0.005 BTC`, no taker/crossing, no `Ioc`, no one-tick-back, public-only waiting before trigger, pre-submit `open_orders` safety, tracked cancel, final open-orders proof, independent open-orders proof, redaction, and T008 ledger fail-closed.
- Added first-class anti-drift artifacts: `anti_drift_gate_manifest.json`, `anti_drift_gate_matrix.csv`, `bbo_stability_matrix.csv`, `adverse_flow_state.csv`, `anti_drift_submit_decision_matrix.csv`, and no-submit reporting.
- Added retry behavior so post-only rejects wait for the next public event and re-run maker-only gates while staying inside the `30` real order endpoint call cap.
- Fixed two fail-closed retry-control issues found during formal runs: post-only reject retry now continues after stale guard, and anti-drift live mode skips stale/failed guard events without consuming a real order attempt.
- Formal run 1 refreshed `awsserver1:/home/admin/hftbacktest-cross-exchange` from `c4faf36b7` to `73d473a48`; remote branch was `cross-exchange`, dirty count `0`, SDK availability `true`, and final gate returned `allow_create_0617T008=true`.
- Formal run 1 evaluated `914` current candidates, anti-drift gate evaluated `99`, passed `6`, blocked `93`, and made `2` real post-only `Alo` buy submissions under the `30` call cap.
- Attempt 1 submitted buy `0.00422 BTC` at `64956.0`, notional `274.11432`, guard/submit BBO `64956/64957`, candidate age `0.575s`, `quality_a`, top-depth multiple `1.85308057`; Hyperliquid rejected it as post-only would-immediately-match after BBO moved to `64954@64955`.
- Attempt 1 latency split was `trigger_to_open_orders_start=0.000210s`, `open_orders_elapsed=0.352091s`, `open_orders_end_to_reprice=0.000133s`, `reprice_to_order_submit=0.000147s`, and exchange response `0.340415s`.
- Attempt 2 submitted buy `0.00179 BTC` at `65032.0`, notional `116.40728`, guard/submit BBO `65032/65033`, candidate age `0.319s`, `quality_a`, top-depth multiple `0.09497207`; Hyperliquid rejected it as post-only would-immediately-match after BBO moved to `65025@65026`.
- Attempt 2 latency split was `trigger_to_open_orders_start=0.000161s`, `open_orders_elapsed=0.033814s`, `open_orders_end_to_reprice=0.000090s`, `reprice_to_order_submit=0.000099s`, and exchange response `0.550789s`.
- Attempt 3 was skipped without order submission because the candidate was stale before order (`candidate_age=2.032s` versus max `1.0s`).
- Formal run 1 produced `fill_count=0`, `maker_fill_count=0`, final open orders empty, independent remote open-orders empty, and T008 returned `live_realized_pnl_proof=false` / `fail_closed_no_realized_live_pnl`.
- Formal rerun 2 refreshed remote from `73d473a48` to `13ec3cce0`; it evaluated `93` current candidates, anti-drift gate evaluated `19`, passed `2`, blocked `17`, made `0` real order submissions, and T008 again failed closed with no realized live PnL.
- Formal rerun 3 refreshed remote from `13ec3cce0` to `5d8a1ec7a`; the SSH session was closed by the remote host before artifact pullback, leaving no watcher artifacts/evaluations in the local rerun2 output.
- After rerun 3 transport interruption, independent remote open-orders checks showed `final_open_orders_empty=true`, the lingering watcher process was terminated, and a follow-up process check showed no remaining watcher process.
- The independent open-orders proof file still carries the legacy `task_id=0622T005` label from `hyperliquid_tiny_live_m2_fill_loop.py`; the proof contents themselves are read-only, final open orders are empty, and no order/cancel endpoint is called by that proof.

verify：
- `python -m pytest examples/hyperliquid/test_hyperliquid_tiny_live_m2_event_driven_watcher.py -q` -> `10 passed`
- `python -m pytest examples/hyperliquid/test_hyperliquid_tiny_live_m2_public_watcher.py -q` -> `4 passed`
- `python -m pytest examples/hyperliquid/test_hyperliquid_tiny_live_m2_fill_loop.py examples/hyperliquid/test_hyperliquid_tiny_live_m2_pnl_ledger.py -q` -> `26 passed`
- `python -m py_compile examples/hyperliquid/hyperliquid_tiny_live_m2_public_watcher.py examples/hyperliquid/hyperliquid_tiny_live_m2_fill_window.py examples/hyperliquid/hyperliquid_tiny_live_m2_fill_loop.py` -> passed
- `python examples/hyperliquid/hyperliquid_tiny_live_m2_public_watcher.py --help` -> passed
- `python examples/hyperliquid/hyperliquid_tiny_live_m2_fill_window.py --help` -> passed
- `python examples/hyperliquid/hyperliquid_tiny_live_m2_fill_loop.py --help` -> passed
- `git diff --check` -> passed before report generation.
- Formal remote refresh, final gate, live anti-drift watcher, pullback, independent open-orders check, cleanup check, and T008 ledger evidence were collected as described above.
- Artifact health: formal run 1 has `121` nonempty artifact rows / `0` empty files in `artifact_nonempty_check.csv`; rerun 2 has `123` nonempty artifact rows / `0` empty files; rerun 3 has only control-plane artifacts because SSH transport ended before watcher artifact pullback.

done：
- T006 implementation is complete and ready for QA.
- The anti-drift gate reduced candidate submissions but did not complete M2. Both real submissions were still rejected by exchange-side post-only validation after fast BBO drift, and no live maker fill / fee / inventory / realized PnL proof exists.
- M2 remains blocked until T008 proves live maker fill plus complete fee / inventory / realized PnL evidence.
- Artifacts are under `local_live_analysis/hyperliquid_tiny_live_m2_anti_drift_gate_0622T006/`, `local_live_analysis/hyperliquid_tiny_live_m2_anti_drift_gate_0622T006_rerun/`, and `local_live_analysis/hyperliquid_tiny_live_m2_anti_drift_gate_0622T006_rerun2/`.

blockers：
- M2 remains blocked: T008 returned `fail_closed_no_realized_live_pnl`.
- The remaining live blocker is not local reprice-to-submit latency; it is exchange-side fast BBO drift / post-only validation plus stale public candidate state around the private `open_orders` gap.
- Rerun 3 also exposed a run-control blocker: SSH transport can close while the remote watcher keeps running. This was cleaned up, with independent proof that final open orders were empty.

commit：
- `73d473a`
- `13ec3cc`
- `5d8a1ec`

提交信息：
- `0622 add anti drift M2 watcher gate`
- `0622 continue anti drift retry after stale guard`
- `0622 keep anti drift live loop after guard skips`
