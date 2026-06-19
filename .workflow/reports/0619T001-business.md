```md
执行线程：
- 业务线程-research

任务ID：
- 0619T001

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0619T001.md`
- `.workflow/reports/0619T001-business.md`
- `examples/hyperliquid/hyperliquid_tiny_live_m2_fill_window.py`
- `examples/hyperliquid/hyperliquid_tiny_live_m2_fill_loop.py`
- `examples/hyperliquid/test_hyperliquid_tiny_live_m2_fill_loop.py`
- `local_live_analysis/hyperliquid_tiny_live_m2_flow_aware_retry_0619T001/**`
- `findings.md`
- `progress.md`
- `task_plan.md`

action：
- Implemented `--side-policy flow_aware` with public-flow precheck, side scoring, crowded-touch filtering, stricter sell handling, shorter `15s` quote hold, quote-aging / adverse-drift early cancel, one tracked order at a time, smaller `0.00999 BTC` cap, and side-score / quote-guard artifacts.
- Fixed remote refresh to use an incremental git bundle after a full bundle upload timed out before any final gate or live order.
- Ran the formal loop after the fix: git-safe refresh succeeded, final gate returned go, public flow precheck passed, and one controlled flow-aware window ran under the approved envelope.

verify：
- `python -m pytest examples/hyperliquid/test_hyperliquid_tiny_live_m2_fill_loop.py examples/hyperliquid/test_hyperliquid_tiny_live_m2_public_flow_diagnosis.py examples/hyperliquid/test_hyperliquid_tiny_live_m2_pnl_ledger.py examples/hyperliquid/test_hyperliquid_tiny_live_real_order_executor.py -q` -> `33 passed`
- `python -m py_compile examples/hyperliquid/hyperliquid_tiny_live_m2_fill_loop.py examples/hyperliquid/hyperliquid_tiny_live_m2_fill_window.py examples/hyperliquid/hyperliquid_tiny_live_m2_public_flow_diagnosis.py examples/hyperliquid/hyperliquid_tiny_live_m2_pnl_ledger.py` -> passed
- `python examples/hyperliquid/hyperliquid_tiny_live_m2_fill_loop.py --help` -> passed
- `python examples/hyperliquid/hyperliquid_tiny_live_m2_fill_window.py --help` -> passed
- Formal loop command -> blocked with `ledger_no_live_realized_pnl_proof`
- JSON validation -> `20` JSON files parsed
- Artifact non-empty check -> `54` files, `0` empty
- Boundary/redaction scan over non-raw artifacts -> `0` hits
- Independent remote open-orders check -> `{'final_open_orders_count': 0, 'final_open_orders_empty': True}`
- `git diff --check` -> passed

done：
- Remote refreshed from `cross-exchange:5391b79439a4f0cb24fd40e4e6aa4b8de53f73e3:0` to `cross-exchange:7d0e813704addc5412e974c79b0eee922075a544:0` using a `60461` byte incremental bundle.
- Final gate returned `tiny_live_ready_for_controller_go`, `allow_create_0617T008=true`, `blocking_reasons=[]`.
- Public-flow precheck collected `20.036s` public data with `l2Book=10`, `trades=19`, `subscription_ack=2`, `reconnects=0`, `close_reason=duration_elapsed`.
- Precheck diagnosis found `6` candidates, `4/6` strict-through, `6/6` touch-trade, `0/6` public top+order depletion, and `4/6` adverse lost-touch.
- Live window completed `6` flow-aware attempts: attempt 1 submitted sell at `62897.0` and reached `resting`; attempt 2 submitted buy at `62880.0` and quote-aging guard canceled/requoted after `lost_touch+adverse_drift`; attempts 3-6 were skipped as `skip_crowded_touch`.
- Window result: `real_order_endpoint_called=true`, `real_cancel_endpoint_called=true`, `post_only_tif=Alo`, `crossing_guard_status=pass`, `shutdown_proof_status=pass`, `final_open_orders_count=0`, `fill_count=0`, `maker_fill_count=0`.
- T008 ledger returned `live_realized_pnl_proof=false` and `realized_pnl_proof_status=fail_closed_no_realized_live_pnl`.
- M2 remains blocked and M3 must not start.

blockers：
- No live maker fill was observed.
- T008 ledger cannot prove fee/rebate, inventory, slippage, mark/PnL, or stable PnL.

commit：
- 083273c
- 7d0e813

提交信息：
- 0619 add M2 flow-aware retry
- 0619 fix M2 incremental remote refresh
```
