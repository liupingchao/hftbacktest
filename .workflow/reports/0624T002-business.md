```md
执行线程：
- 业务线程-live-awsserver1

任务ID：
- 0624T002

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0624T002.md`
- `.workflow/reports/0624T002-business.md`
- `examples/hyperliquid/hyperliquid_tiny_live_m2_public_watcher.py`
- `examples/hyperliquid/test_hyperliquid_tiny_live_m2_event_driven_watcher.py`
- `local_live_analysis/hyperliquid_tiny_live_m2_bbo_evidence_chain_repair_0624T002/t010_replay_repair_validation/**`
- `task_plan.md`
- `progress.md`
- `findings.md`

action：
- Added live candidate evidence fields for BBO history/cache visibility, same-touch queue-reset deltas, fresh-touch block taxonomy, queue-reset block taxonomy, and local/exchange ordering status.
- Extended BBO history retention from the prior short anti-drift window to `60000ms` while keeping latest-L2 stale evidence fail-closed at `BBO_HISTORY_STALE_MS=5000`.
- Added `--generate-bbo-evidence-chain-repair-validation`.
- Added public-only T010 replay validation outputs:
  - `bbo_candidate_evidence_repaired.csv`
  - `bbo_repair_reason_taxonomy.csv`
  - `bbo_repair_summary.csv`
  - `bbo_repair_manifest.json`
  - `README.md`
- Kept same-touch evidence conservative by using the current continuous same-touch segment; no cross-touch stitching is used as pass evidence.
- Did not change quote distance, cap, post-only behavior, private/order boundaries, or accepted pass criteria. `synthetic_current_event_only` remains fail-closed.

artifact：
- Local output:
  - `local_live_analysis/hyperliquid_tiny_live_m2_bbo_evidence_chain_repair_0624T002/t010_replay_repair_validation/`
- Source artifact replayed:
  - `local_live_analysis/hyperliquid_tiny_live_m2_aws_candidate_funnel_0623T010_20260623T064432Z/venv_public_shadow_soak/`

T010 replay repair result：
- `candidate_count=1259`
- `stream_total_book_event_count=112`
- `stream_total_trade_event_count=3449`
- `required_repaired_fields_present=true`
- `repaired_fresh_touch_evidence_pass_count=1068`
- `repaired_synthetic_current_event_only_count=6`
- `bbo_history_too_sparse_count=6`
- `same_touch_stable_enough_count=1068`
- `same_touch_reset_supported_count=317`
- `history_present_no_reset_count=165`
- `local_receive_ordering_ok_count=1259`
- `exchange_time_ordering_conflict_count=1`
- `dominant_blocker_after_repair=bbo_evidence_repaired_remaining_blocker_is_flow_or_downstream_gate`

reason taxonomy：
- BBO history status:
  - `same_touch_stable_enough=1068`
  - `same_touch_seen_but_not_stable=102`
  - `last_l2_too_old=83`
  - `bbo_history_too_sparse=6`
- Fresh-touch evidence:
  - `pass=1068`
  - `block=191`
- Fresh-touch block reasons:
  - `same_touch_seen_but_not_stable=102`
  - `last_l2_too_old=83`
  - `bbo_history_too_sparse=6`
- Queue-reset evidence:
  - `reset_supported=317`
  - `history_present_no_reset=165`
  - `no_prior_same_touch_bbo=771`
  - `bbo_history_too_sparse=6`
- Ordering:
  - `local_receive_ordering_status=latest_l2_received_before_or_at_candidate` for `1259/1259`
  - `exchange_time_ordering_conflict_count=1`

conclusion：
- The repaired evidence-chain shows the dominant T001/T010 blocker was not a lack of Binance freshness, fair-mid, or edge signal. It was the public-shadow candidate evidence/cache accounting around BBO history.
- After reconstructing decision-time-visible BBO history by local receive order, `synthetic_current_event_only` drops from `1257` in the original T001 diagnosis to `6` in repaired validation, while `1068` candidates have accepted same-touch stability evidence and `317` have same-touch reset support.
- Remaining blockers are now better classified as flow/downstream gates or explicit BBO evidence subcases (`same_touch_seen_but_not_stable`, `last_l2_too_old`, `bbo_history_too_sparse`), not as a generic missing BBO-history bucket.
- A fresh AWS public-only no-submit rerun was not needed because the T010 row-level replay had enough l2Book/trade/candidate evidence to validate required fields and the repaired taxonomy. The new live candidate audit fields will be populated in any later AWS public-only run.

verify：
- `python -m pytest examples/hyperliquid/test_hyperliquid_tiny_live_m2_event_driven_watcher.py -q` -> `37 passed`
- `python -m py_compile examples/hyperliquid/hyperliquid_tiny_live_m2_public_watcher.py` -> passed
- `python examples/hyperliquid/hyperliquid_tiny_live_m2_public_watcher.py --help` -> passed
- `python examples/hyperliquid/hyperliquid_tiny_live_m2_public_watcher.py --generate-bbo-evidence-chain-repair-validation --shadow-output-dir local_live_analysis/hyperliquid_tiny_live_m2_aws_candidate_funnel_0623T010_20260623T064432Z/venv_public_shadow_soak --output-dir local_live_analysis/hyperliquid_tiny_live_m2_bbo_evidence_chain_repair_0624T002/t010_replay_repair_validation --artifact-task-id 0624T002` -> passed
- JSON validation for `bbo_repair_manifest.json` -> passed
- Empty artifact check -> passed, no empty files
- CSV line counts -> `1302` total lines across repair validation CSV outputs
- `git diff --check` -> passed

boundary：
- `real_orders_allowed=false`
- `next_real_canary_authorized=false`
- `credential_reads_allowed=false`
- `private_or_order_endpoint_allowed=false`
- `quote_distance_changed=false`
- `cap_relaxation=false`
- `fresh_touch_requirements_weakened=false`
- No live orders, no credential reads, no private/account/order/cancel endpoints, no remote final gate rerun, no T008 live ledger claim, no one-tick-back, no inside-spread, no taker/crossing, no M3/stable PnL/default-on/promotion.

done：
- `0624T002` completed the requested public BBO evidence-chain repair plus public-only T010 replay validation.
- Result is ready for QA.

blockers：
- No execution blocker.
- Forward blocker: this task does not authorize live orders or canary. Next controller decision should focus on whether to run a fresh AWS public-only no-submit validation with the new fields, or proceed to downstream flow/fair-mid/edge funnel validation using repaired BBO evidence.

commit：
- 29870c4

提交信息：
- 0624 repair BBO evidence chain diagnostics
```
