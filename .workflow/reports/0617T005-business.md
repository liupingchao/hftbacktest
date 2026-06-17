# 0617T005 Business Report

执行线程：
- 业务线程-research

任务ID：
- 0617T005

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0617T005.md`
- `.workflow/reports/0617T005-business.md`
- `docs/hyperliquid_tiny_live_signal_quote_replay.md`
- `examples/hyperliquid/hyperliquid_tiny_live_signal_quote_replay.py`
- `examples/hyperliquid/test_hyperliquid_tiny_live_signal_quote_replay.py`
- `local_live_analysis/hyperliquid_tiny_live_signal_quote_replay_0617T005/**`
- `task_plan.md`
- `progress.md`
- `findings.md`

action：
- QA accepted `0617T004` and copied the latest QA result to `docs/qa-acceptance-report.md`.
- Formally dispatched `0617T005` after `0617T004` QA passed.
- Implemented a read-only signal / quote replay runner for the `0617T004` protocol.
- Fixed local source resolution for `0609T008` manifest paths so old absolute paths relocate under this host's `local_live_analysis/` tree.
- Ran full quote replay on `8` locally available pricing-signal inputs: `0601T005` plus all `7` historical event-mode samples from `0609T008`.
- Replayed `161455` raw pricing rows and evaluated `26948` de-duplicated decision rows across threshold grid `10,20,30,40,50,75,100` ticks and persistence grid `1,2,3`.
- Wrote `calibration_summary.csv` with threshold-level intent rates, stale/data-gap rates, cap/reduce-side-only rates, and sample coverage.
- Selected `75` ticks with `2` observations as the primary read-only candidate, with `75` ticks and `3` observations as a stricter low-activity fallback.

final recommendation：
- `hyperliquid_tiny_live_signal_quote_replay_ready_for_qa`

verify：
- `/home/liushuai/workspace/hftbacktest/.conda-envs/hft-py38/bin/python --version` could not run on this host because that path does not exist.
- `python --version` -> `Python 3.13.5` at `/home/molly/anaconda3/bin/python`.
- `python -m pytest examples/hyperliquid/test_hyperliquid_tiny_live_signal_quote_replay.py -q` passed.
- `python examples/hyperliquid/hyperliquid_tiny_live_signal_quote_replay.py` passed.
- `python -m json.tool local_live_analysis/hyperliquid_tiny_live_signal_quote_replay_0617T005/replay_manifest.json` passed.
- Required CSV / markdown artifacts are non-empty.
- Boundary review passed: no private endpoint, no account query, no credential read, no order placement, no cancellation, no amendment, no live bot, no PnL claim, no fill claim, no maker viability claim.
- `git diff --check` passed.

done：
- Full quote replay inputs: `8` local `pricing_signal_rows.csv` files listed in `local_live_analysis/hyperliquid_tiny_live_signal_quote_replay_0617T005/replay_manifest.json`.
- Source availability output confirms all `7` `0609T008` historical manifest samples are `local_direct_file_available=true` and `replay_source_used=pricing_signal_rows`.
- Primary candidate: threshold `75` ticks, persistence `2`, `654` accepted theoretical intents (`313` buy / `341` sell), `2.4269%` intent rate, `8/8` samples with any intent.
- Stricter fallback: threshold `75` ticks, persistence `3`, `327` accepted theoretical intents (`148` buy / `179` sell), `1.2134%` intent rate, `8/8` samples with any intent.
- `100` ticks with persistence `3` is too sparse for balanced bootstrap evidence: `193` intents and only `5/8` samples with buy intent.
- Historical threshold distribution input: `local_live_analysis/basis_positive_row_level_generator_0609T008/row_level_read_only_cases.csv`.
- Source availability output: `local_live_analysis/hyperliquid_tiny_live_signal_quote_replay_0617T005/source_availability.csv`.
- Threshold sensitivity output: `local_live_analysis/hyperliquid_tiny_live_signal_quote_replay_0617T005/threshold_sensitivity.csv`.
- Calibration summary output: `local_live_analysis/hyperliquid_tiny_live_signal_quote_replay_0617T005/calibration_summary.csv`.
- Row-level basis distribution output: `local_live_analysis/hyperliquid_tiny_live_signal_quote_replay_0617T005/row_level_basis_distribution_by_sample.csv`.
- Row-level audit output: `local_live_analysis/hyperliquid_tiny_live_signal_quote_replay_0617T005/row_level_audit_sample.csv`.
- Result: `0617T005` read-only replay evidence is ready for QA. It still does not authorize real orders or `0616T008` live execution without QA/controller ratification.
- No credentials were read; no private API was called; no account query occurred; no orders were placed/cancelled/amended; no live bot was started.

blockers：
- 无

commit：
- 34dd3ce

提交信息：
- 0617 multi-sample signal quote replay calibration
