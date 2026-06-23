```md
执行线程：
- 业务线程-live-awsserver1

任务ID：
- 0623T010

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0623T010.md`
- `.workflow/reports/0623T010-business.md`
- `local_live_analysis/hyperliquid_tiny_live_m2_aws_candidate_funnel_0623T010_20260623T064432Z/**`
- `task_plan.md`
- `progress.md`
- `findings.md`

action：
- Created `0623T010` as the AWS public-only candidate funnel diagnosis task.
- Ran a 600s public-only no-submit shadow window on `awsserver1` with `/home/admin/.venvs/hyperliquid-sdk-0618T002/bin/python`.
- Pulled remote artifacts back to local and generated `candidate_funnel_summary/**` from the row-level CSV outputs.
- The task did not change quote distance, caps, post-only policy, live order behavior, credential handling, private/account/order endpoints, or final gate state.

remote / local artifacts：
- Remote output: `/home/admin/hftbacktest-cross-exchange-artifacts/hyperliquid_tiny_live_m2_aws_candidate_funnel_0623T010_20260623T064432Z/`.
- Local output: `local_live_analysis/hyperliquid_tiny_live_m2_aws_candidate_funnel_0623T010_20260623T064432Z/`.
- Funnel summary: `local_live_analysis/hyperliquid_tiny_live_m2_aws_candidate_funnel_0623T010_20260623T064432Z/candidate_funnel_summary/`.

runtime evidence：
- The 600s live public shadow run completed with `duration_elapsed`.
- Public stream counts: `l2Book=112`, `trades=1147`, `subscription_ack=2`, `reconnects=0`, `total_trade_event_count=3449`.
- Public candidate rows: `current_candidate_count=1259`, `shadow_evaluation_count=1259`.
- Shadow result stayed fail-closed: `shadow_would_submit_count=0`, `fair_mid_source_pass_count=0`, `fair_mid_source_block_count=0`, `edge_gate_pass_count=0`, `edge_gate_block_count=0`, `source_path_exercised=false`.

candidate funnel diagnosis：
- First blocking stage: `fresh_touch_gate_allowed`.
- `public_events_evaluated=1259`.
- `fresh_touch_evidence_pass=2` (`0.158856%` of public events).
- `strict_trade_through_seen=299` (`23.749007%`).
- `at_or_through_trade_seen=1054` (`83.717236%`).
- `visible_top_plus_order_depleted=129` (`10.246227%`).
- `fresh_touch_gate_allowed=0`.
- Because no candidate passed the accepted fresh-touch / dynamic-size gate, anti-drift, Binance freshness, fair-mid source, and edge gate were never reached.
- Dominant skip atoms:
  - `missing_touch_freshness_or_queue_reset_evidence=1257`
  - `missing_same_side_strict_through_support=960`
  - `missing_recent_same_side_at_or_through_throughput=205`
- Dominant skip combinations:
  - `missing_same_side_strict_through_support;missing_touch_freshness_or_queue_reset_evidence=753`
  - `missing_touch_freshness_or_queue_reset_evidence=299`
  - `missing_same_side_strict_through_support;missing_touch_freshness_or_queue_reset_evidence;missing_recent_same_side_at_or_through_throughput=205`
  - `missing_same_side_strict_through_support=2`

boundary：
- `credentials_read=false`
- `private_endpoint_called=false`
- `account_endpoint_called=false`
- `order_endpoint_called=false`
- `cancel_endpoint_called=false`
- `live_client_initialized=false`
- `real_orders_allowed=false`
- `no_submit_enforced=true`
- `next_real_canary_authorized=false`

verify：
- `python -m pytest examples/hyperliquid/test_hyperliquid_tiny_live_m2_event_driven_watcher.py -q` -> `35 passed`
- `python -m py_compile examples/hyperliquid/hyperliquid_tiny_live_m2_public_watcher.py` -> passed
- `python examples/hyperliquid/hyperliquid_tiny_live_m2_public_watcher.py --help` -> passed
- Remote JSON validation for `public_shadow_source_manifest.json` and `boundary_manifest.json` -> passed
- Remote empty-file check -> passed
- Remote line counts: `current_candidate_audit.csv=1260`, `public_shadow_decision_matrix.csv=1260`, `fair_mid_source_matrix.csv=1`, `edge_gate_matrix.csv=1`, `public_source_freshness_matrix.csv=1`, `rolling_flow_state.csv=1260`
- Local JSON validation for `public_shadow_source_manifest.json` and `boundary_manifest.json` -> passed
- Local empty-file check -> passed
- Local line counts matched remote

done：
- T010 completed the requested public-only candidate funnel diagnosis.
- The primary blocker is now localized before Binance freshness / fair-mid source / edge gate: no candidate passed the accepted fresh-touch / dynamic-size gate.
- This means the next useful work should inspect or repair fresh-touch evidence generation / BBO-history continuity / queue-reset recognition under the public shadow path, not loosen quote distance, caps, post-only, private/order boundaries, or canary authorization.
- The next real canary remains unauthorized.

blockers：
- No task-scoped execution blocker.
- Forward blocker: `fresh_touch_gate_allowed=0`; fair-mid and edge gates cannot be evaluated until a candidate reaches them.

commit：
- 813158d

提交信息：
- 0623 run aws public candidate funnel
```
