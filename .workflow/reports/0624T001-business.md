```md
执行线程：
- 业务线程-live-awsserver1

任务ID：
- 0624T001

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0624T001.md`
- `.workflow/reports/0624T001-business.md`
- `examples/hyperliquid/hyperliquid_tiny_live_m2_public_watcher.py`
- `examples/hyperliquid/test_hyperliquid_tiny_live_m2_event_driven_watcher.py`
- `local_live_analysis/hyperliquid_tiny_live_m2_aws_bbo_evidence_chain_0624T001/t010_replay_bbo_evidence_chain/**`
- `task_plan.md`
- `progress.md`
- `findings.md`

action：
- Added an offline public-only BBO evidence-chain diagnosis mode:
  - `python examples/hyperliquid/hyperliquid_tiny_live_m2_public_watcher.py --generate-bbo-evidence-chain-diagnosis`
- Added diagnosis outputs for:
  - BBO evidence summary
  - freshness / reset / skip-reason histograms
  - l2Book/trade density by minute
  - event-ordering matrix with previous/next l2Book/trade context
  - representative rejected candidate examples
  - manifest / README
- Replayed the existing AWS `0623T010` public-shadow artifacts instead of opening a fresh live public window, because T010 already had sufficient row-level candidate, stream, and decision evidence for this task.
- Preserved all public-only / no-submit boundaries and did not change any fresh-touch / quote / cap / post-only decision rule.

artifact：
- Local output:
  - `local_live_analysis/hyperliquid_tiny_live_m2_aws_bbo_evidence_chain_0624T001/t010_replay_bbo_evidence_chain/`
- Source AWS artifact replayed:
  - `local_live_analysis/hyperliquid_tiny_live_m2_aws_candidate_funnel_0623T010_20260623T064432Z/venv_public_shadow_soak/`

diagnosis result：
- `candidate_count=1259`
- `decision_row_count=1259`
- `stream_total_book_event_count=112`
- `stream_total_trade_event_count=3449`
- `l2book_candidate_count=112`
- `trade_candidate_count=1147`
- `book_to_trade_event_ratio_pct=3.247318`
- `synthetic_current_event_only_count=1257`
- `synthetic_current_event_only_share_pct=99.841144`
- `fresh_touch_evidence_pass_count=2`
- `fresh_touch_allowed_count=0`
- `queue_reset_supported_count=2`
- `strict_trade_through_seen_count=299`
- `at_or_through_trade_seen_count=1054`
- `visible_top_plus_order_depleted_count=139`
- `exchange_time_regression_count=1`
- `trade_older_than_latest_l2_count=1`
- `negative_next_l2_delta_count=0`
- `dominant_blocker_classification=public_bbo_density_or_cache_continuity_blocks_bbo_history_visibility`

conclusion：
- The main blocker is not Binance freshness, fair-mid source, or edge gate. Those stages still are not reached because `fresh_touch_allowed_count=0`.
- The dominant evidence-chain issue is that the public-shadow path mostly evaluates trade-triggered candidates while accepted real BBO-history evidence is almost always unavailable.
- Event ordering anomalies exist but are rare in this sample, so they are not the dominant blocker.
- The next repair should inspect live public book subscription/update handling and BBO history/cache construction under the public-shadow path.
- Do not fix this by relaxing quote distance, cap, post-only behavior, fresh-touch evidence requirements, or private/order boundaries.

verify：
- `python -m pytest examples/hyperliquid/test_hyperliquid_tiny_live_m2_event_driven_watcher.py -q` -> `36 passed`
- `python -m py_compile examples/hyperliquid/hyperliquid_tiny_live_m2_public_watcher.py` -> passed
- `python examples/hyperliquid/hyperliquid_tiny_live_m2_public_watcher.py --help` -> passed
- `python examples/hyperliquid/hyperliquid_tiny_live_m2_public_watcher.py --generate-bbo-evidence-chain-diagnosis --shadow-output-dir local_live_analysis/hyperliquid_tiny_live_m2_aws_candidate_funnel_0623T010_20260623T064432Z/venv_public_shadow_soak --output-dir local_live_analysis/hyperliquid_tiny_live_m2_aws_bbo_evidence_chain_0624T001/t010_replay_bbo_evidence_chain --artifact-task-id 0624T001` -> passed
- JSON validation for `bbo_evidence_chain_manifest.json` -> passed
- Empty artifact check -> passed, no empty files
- CSV line counts -> `1329` total lines across diagnosis CSV outputs
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
- 0624T001 completed the requested public BBO evidence-chain diagnosis and produced a reusable offline diagnostic mode plus T010 replay artifacts.
- Result is ready for QA.

blockers：
- No execution blocker.
- Forward blocker: public-shadow fresh-touch evidence remains blocked by public BBO density / BBO-history cache visibility; no candidate reaches Binance freshness, fair-mid source, or edge gate.

commit：
- e61534c

提交信息：
- 0624 add BBO evidence chain diagnosis
```
