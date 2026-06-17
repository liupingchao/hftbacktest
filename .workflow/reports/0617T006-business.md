# 0617T006 Business Report

执行线程：
- 业务线程-research

任务ID：
- 0617T006

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0617T006.md`
- `.workflow/reports/0617T006-business.md`
- `docs/hyperliquid_tiny_live_optimistic_pnl_proxy.md`
- `examples/hyperliquid/hyperliquid_tiny_live_optimistic_pnl_proxy.py`
- `examples/hyperliquid/test_hyperliquid_tiny_live_optimistic_pnl_proxy.py`
- `local_live_analysis/hyperliquid_tiny_live_optimistic_pnl_proxy_0617T006/**`
- `task_plan.md`
- `progress.md`
- `findings.md`

action：
- `0617T005` QA passed and was copied to `docs/qa-acceptance-report.md` before this execution.
- Implemented a read-only optimistic PnL proxy runner over local `pricing_signal_rows.csv` artifacts accepted or consumed by `0617T005`.
- Reconciled the user's `6 datasets` wording against accepted local manifests before computing PnL.
- After user clarification, selected `canonical_7` as the formal sample-set口径.
- Recorded `requested_six` as superseded by user-selected `canonical_7`.
- Computed the official estimate for `canonical_7` and diagnostic comparison estimate for `0617T005_8_input`.
- Used the `0617T004` / `0617T005` side mapping and threshold candidates: positive eligible signal -> Hyperliquid maker buy intent, negative eligible signal -> Hyperliquid maker sell intent, primary `75` ticks / persistence `2`, fallback `75` ticks / persistence `3`, sensitivity grid `50,75,100` x `1,2,3`.
- Computed fixed-horizon and non-tradeable oracle-best-horizon optimistic PnL proxy under `unconstrained_all_intents`.

final recommendation：
- `hyperliquid_tiny_live_optimistic_pnl_proxy_ready_for_qa`

verify：
- `/home/liushuai/workspace/hftbacktest/.conda-envs/hft-py38/bin/python --version` could not run on this host because that path does not exist.
- `python --version` -> `Python 3.13.5` at `/home/molly/anaconda3/bin/python`.
- `python -m pytest examples/hyperliquid/test_hyperliquid_tiny_live_optimistic_pnl_proxy.py -q` passed.
- `python examples/hyperliquid/hyperliquid_tiny_live_optimistic_pnl_proxy.py` passed.
- `python -m json.tool local_live_analysis/hyperliquid_tiny_live_optimistic_pnl_proxy_0617T006/optimistic_pnl_proxy_manifest.json` passed.
- Required CSV / markdown artifacts are non-empty.
- Boundary review passed: no private endpoint, no account query, no credential read, no order placement, no cancellation, no amendment, no live bot, no fill-probability model, no queue-priority model, no real PnL claim, no real fill claim, no maker viability claim, no live authorization.
- `git diff --check` passed.

done：
- Input sample-set reconciliation output: `local_live_analysis/hyperliquid_tiny_live_optimistic_pnl_proxy_0617T006/sample_set_reconciliation.csv`.
- `requested_six`: superseded by user-selected `canonical_7`.
- `canonical_7`: official sample set, computed from `7` canonical event-mode `0609T008` manifest samples, `139914` pricing rows, `23353` decision rows.
- `0617T005_8_input`: computed from all `8` accepted `0617T005` pricing inputs, `161455` pricing rows, `26948` decision rows.
- Formula: buy = future Hyperliquid mid move ticks + half spread; sell = negative future Hyperliquid mid move ticks + half spread; USDC = ticks * `0.1` tick size * `0.01 BTC`.
- Primary fixed-horizon diagnostic at `75` ticks / persistence `2` / `1000ms`: `canonical_7` has `7231` rows, `295.985 USDC` optimistic proxy, mean `40.932789` ticks per intent, and `7/7` samples positive.
- Primary fixed-horizon diagnostic at `75` ticks / persistence `2` / `1000ms`: `0617T005_8_input` has `7597` rows, `299.38 USDC` optimistic proxy, mean `39.407661` ticks per intent, and `8/8` samples positive.
- Oracle-best-horizon diagnostic at `75` ticks / persistence `2`: `canonical_7` has `7238` intents and `1098.535 USDC` optimistic proxy.
- Oracle-best-horizon diagnostic at `75` ticks / persistence `2`: `0617T005_8_input` has `7604` intents and `1124.89 USDC` optimistic proxy.
- Fixed-horizon output: `local_live_analysis/hyperliquid_tiny_live_optimistic_pnl_proxy_0617T006/fixed_horizon_pnl_summary.csv`.
- Aggregate fixed-horizon output: `local_live_analysis/hyperliquid_tiny_live_optimistic_pnl_proxy_0617T006/aggregate_fixed_horizon_pnl_summary.csv`.
- Oracle output: `local_live_analysis/hyperliquid_tiny_live_optimistic_pnl_proxy_0617T006/oracle_best_horizon_summary.csv`.
- Row-level audit output: `local_live_analysis/hyperliquid_tiny_live_optimistic_pnl_proxy_0617T006/row_level_audit_sample.csv`.
- This is an optimistic public-data upper bound only. It is not成交概率、真实手续费、真实 PnL、账户库存、execution quality、maker viability 或 live authorization.
- No credentials were read; no private API was called; no account query occurred; no orders were placed/cancelled/amended; no live bot was started.

blockers：
- 无

commit：
- d407dab

提交信息：
- 0617 select canonical pnl proxy sample set
