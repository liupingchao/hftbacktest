# 0706T008 Business Report

执行线程：
- 业务线程-live-awsserver1

任务ID：
- 0706T008

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0706T008.md`
- `.workflow/reports/0706T008-business.md`
- `docs/cross_exchange_t010_candidate_live_auto_loop_plan.md`
- `local_live_analysis/cross_exchange_t010_long_window_nosubmit_0706T008_20260706T102343Z/**`

action：
- Created the three-task auto-loop plan for long-window no-submit diagnosis -> repair or controlled live evidence.
- Committed the plan and task dispatch node before execution.
- Synced `awsserver1:/home/admin/hftbacktest-cross-exchange` to commit `8a31a769b5b33aa2a2930b0eccc96777e010b07e` with a fast-forward git bundle workflow.
- Ran a `1800s` AWS public-only/no-submit shadow window using `/home/admin/.venvs/hyperliquid-sdk-0618T002/bin/python`.
- Generated BBO evidence-chain repair validation and BBO/funnel diagnosis from the same fresh window.
- Pulled remote artifacts back to local.

remote / local artifacts：
- Remote output:
  - `/home/admin/hftbacktest-cross-exchange-artifacts/cross_exchange_t010_long_window_nosubmit_0706T008_20260706T102343Z/`
- Local output:
  - `local_live_analysis/cross_exchange_t010_long_window_nosubmit_0706T008_20260706T102343Z/`

result：
- Route recommendation: `route_to_controlled_live_evidence_task`.
- Task 2 repair is not recommended from this result.
- Task 3 controlled live evidence is justified under the plan because no-submit evidence produced an eligible would-submit path.
- Window elapsed: `1800.001312s`.
- Public stream:
  - l2Book messages: `336`
  - trades messages: `2144`
  - total book events: `336`
  - total trade events: `6881`
  - subscription ack count: `2`
  - reconnect count: `0`
  - public timeout count: `5`
- Candidate/funnel:
  - current candidates: `2480`
  - shadow evaluations: `2480`
  - fresh-touch evidence pass: `2170`
  - fresh-touch allowed: `125`
  - anti-drift pass/block: `7` / `118`
  - fair-mid source pass/block: `1` / `6`
  - edge gate pass/block: `1` / `6`
  - shadow would-submit: `1`
- Would-submit row:
  - event sequence: `1369`
  - source channel: `trades`
  - side: `buy`
  - quote: `63019`
  - bid/ask: `63019` / `63020`
  - spread: `1` tick
  - order size: `0.005 BTC`
  - quality bucket: `quality_a`
  - fair-mid source age: `43ms`
  - fair-mid: `63044.5`
  - edge: `25.5` ticks
  - shadow action: `would_submit_if_real_order_task_authorized`
  - shadow reason: `edge_gate_pass_shadow_no_submit`

boundary：
- `credentials_read=false`
- `private_endpoint_called=false`
- `account_endpoint_called=false`
- `order_endpoint_called=false`
- `cancel_endpoint_called=false`
- `live_client_initialized=false`
- `no_submit_enforced=true`
- `real_orders_allowed=false`
- `no_final_gate_rerun=true`
- No live order was submitted.

verify：
- `python -m pytest examples/hyperliquid/test_hyperliquid_tiny_live_m2_event_driven_watcher.py -q` -> `37 passed`
- `python -m py_compile examples/hyperliquid/hyperliquid_tiny_live_m2_public_watcher.py` -> passed
- `python examples/hyperliquid/hyperliquid_tiny_live_m2_public_watcher.py --help` -> passed
- JSON parse and CSV schema checks passed.
- Boundary manifest checked.
- Empty-file check passed.
- Redaction scan found no raw secret/private key/signature values; matches were expected field names and boundary text.
- `git diff --check` pending final QA step.

done：
- Long-window no-submit diagnosis completed.
- The result says the 20s `0706T007` no-submit outcome was too short to conclude candidate absence.
- The current route should skip repair and create the controlled live evidence task `0706T010` under the already planned low-risk envelope.

blockers：
- Full `0625T010` remains blocked until controlled live evidence produces submitted lifecycle/economics/open-orders evidence and a later replay acceptance task validates it.

commit：
- 8a31a76

提交信息：
- Plan T010 candidate live evidence auto loop
