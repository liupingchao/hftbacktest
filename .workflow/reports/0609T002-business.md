# 0609T002 业务线程报告

执行线程：
- 业务线程-research

任务ID：
- 0609T002

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0609T002.md`
- `.workflow/reports/0609T002-business.md`
- `local_live_analysis/basis_positive_targeted_public_collection_0609T002/`
- `local_live_analysis/canonical_basis_positive_wrong_way_decomposition_0609T002/`

action：
- 在 `0609T001` QA 通过且结论为 `targeted_collection_ready` 后，使用 `awsserver1` task-scoped clean worktree `/home/admin/hft_live/worktrees/0609T002-basis-positive-targeted-public` 执行 public-only synchronized collection。
- 远端采集 commit 为 `f25d826`，采集命令统一使用 `/home/admin/hft_live/venv/bin/python examples/hyperliquid/synchronized_public_collection.py collect --duration-seconds 1800 --task-id 0609T002 --clean-output`，输出根目录为 `/home/admin/hft_live/runs/0609T002-basis-positive-targeted-public/`。
- 完成 4 个 1800s public-only 样本：`xemm_0609_normal_a`、`xemm_0609_normal_b`、`xemm_0609_active_a`、`xemm_0609_active_b`，远端 exit code 全部为 `0`。
- 将 public raw artifacts 拉回 local machine 后，从 pulled raw artifacts 本地重建 Binance sidecar/provenance、Hyperliquid event-mode alignment、as-of join、lead-lag analysis、pricing-signal artifacts、7-sample event-mode canonical aggregate，并重跑 T001-style basis-positive wrong-way decomposition。

remote collection：
- `xemm_0609_normal_a`: `2026-06-09T02:40:38+09:00` -> `2026-06-09T03:14:23+09:00`, overlap `1800.017152859s`, Binance depth/bookTicker/trade `66705/771081/86873`, Hyperliquid `l2Book=3342`, classification `passes_pricing_research_market_view`。
- `xemm_0609_normal_b`: `2026-06-09T03:14:23+09:00` -> `2026-06-09T03:47:42+09:00`, overlap `1800.002481473s`, Binance depth/bookTicker/trade `66566/665884/63351`, Hyperliquid `l2Book=3341`, classification `passes_pricing_research_market_view`。
- `xemm_0609_active_a`: `2026-06-09T03:47:42+09:00` -> `2026-06-09T04:21:03+09:00`, overlap `1800.003859020s`, Binance depth/bookTicker/trade `66575/635889/60572`, Hyperliquid `l2Book=3339`, classification `passes_pricing_research_market_view`。
- `xemm_0609_active_b`: `2026-06-09T04:21:03+09:00` -> `2026-06-09T04:54:34+09:00`, overlap `1800.070242694s`, Binance depth/bookTicker/trade `66609/796179/67114`, Hyperliquid `l2Book=3342`, classification `passes_pricing_research_market_view`。
- Remote tmux follow-up check: `ssh admin@awsserver1 tmux ls` returned no tmux server, so `0609T002_collect` was no longer running.

pullback / local processing：
- `pulled_artifact_manifest.json` reports `all_sha256_match=true`; all Binance and Hyperliquid raw sha256 checks match recorded values.
- `sample_quality_matrix.csv` confirms all 4 new samples have `remote_exit_code=0`, `passes_target_1800s=True`, `binance_depth_pu_mismatch_count=0`, `hyperliquid_decision_mode=event`, and Hyperliquid classification `passes_pricing_research_market_view`.
- `local_processing_manifest.json` records accepted local processing from pulled `awsserver1` public raw artifacts and reports `aggregate_canonical_sample_count=7`, `decomposition_canonical_sample_count=7`, and task-level final recommendation `tail_filter_hypothesis_validated_for_read_only_research`。
- The refreshed canonical aggregate combines the prior 3 accepted event-mode canonical samples with the 4 new T002 samples; `pricing_signal_robustness_recommendation.md` remains `continue_read_only_runner_refinement` for read-only runner evidence.

decomposition result：
- Refreshed decomposition output: `local_live_analysis/canonical_basis_positive_wrong_way_decomposition_0609T002/`.
- `basis > 0`: `5726` rows across `7` samples, hit rate `0.96462347`, mean future move `33.71201537` ticks, wrong-way count `101`, wrong-way rate `0.01763884`, p95 wrong-way loss `120` ticks, max wrong-way loss `180` ticks, max sample row share `0.28763535`。
- `basis <= 0`: `17616` rows across `7` samples, hit rate `0.29276644`, mean future move `-11.82078792` ticks, wrong-way count `2052`, max sample row share `0.17018619`。
- Positive-basis magnitude remains monotonic by mean future move: small `12.74463632`, medium `23.66057441`, large `64.93157895` ticks.
- Controlled support remains true for Binance momentum and Hyperliquid book state.
- Validated read-only visible tail hypotheses:
  - `basis_magnitude_bucket=basis_positive_small`: wrong-way `51`, sample count `7`, classification `promising_visible_filter`。
  - `hl_top5_imbalance_bucket=hl_top5_imbalance_negative_small`: wrong-way `35`, sample count `7`, classification `promising_visible_filter`。
  - `hl_microprice_minus_mid_bucket=hl_microprice_minus_mid_negative_small`: wrong-way `33`, sample count `7`, classification `promising_visible_filter`。

final recommendation：
- `tail_filter_hypothesis_validated_for_read_only_research`
- 含义：T001 提出的 visible tail-filter hypotheses 已在 7-sample read-only public evidence 中重复出现，且 basis-positive sample concentration 降到 `0.40` gate 以下。
- 这只是 public-data read-only research 结论；不授权 strategy implementation、private/account/order endpoints、order lifecycle、case-library implementation、shadow decision generation、live/default-on/tiny-live、parameter search、deployment recommendation 或 promotion。

verify：
- `ssh admin@awsserver1 tmux ls` -> no server running, no residual remote tmux collection session.
- `ps -eo pid,ppid,stat,etime,cmd | rg '0609T002|basis_positive_targeted|synchronized_public_collection|canonical_basis_positive'` -> only the check command itself matched; no residual local T002 long process.
- `python -m json.tool local_live_analysis/basis_positive_targeted_public_collection_0609T002/remote_collection_manifest.json` -> passed.
- `python -m json.tool local_live_analysis/basis_positive_targeted_public_collection_0609T002/pulled_artifact_manifest.json` -> passed.
- `python -m json.tool local_live_analysis/basis_positive_targeted_public_collection_0609T002/local_processing_manifest.json` -> passed.
- `python -m json.tool local_live_analysis/canonical_basis_positive_wrong_way_decomposition_0609T002/basis_positive_wrong_way_manifest.json` -> passed.
- Full task artifact parse: `125` JSON files parsed with `json_bad=[]`; `123` CSV files parsed with `csv_bad=[]`.
- `python examples/hyperliquid/synchronized_public_collection.py --help` -> passed.
- `python examples/hyperliquid/synchronized_public_collection.py collect --help` -> passed.
- `python examples/hyperliquid/canonical_basis_positive_wrong_way_decomposition.py --help` -> passed.
- `python -m pytest examples/hyperliquid/test_synchronized_public_collection.py examples/hyperliquid/test_canonical_basis_positive_wrong_way_decomposition.py -q` -> `13 passed in 0.11s`.
- Boundary text check found only prohibition/scope/boundary statements for private/order/strategy/live/default-on/tiny-live/case-library/shadow/promotion/parameter terms.
- `git diff --check` -> passed.

done：
- Remote collection, pullback, local event-mode processing, refreshed 7-sample aggregate, and refreshed wrong-way decomposition are complete.
- All accepted raw collection happened on `awsserver1`; all accepted formal analysis/testing happened on the local machine from pulled public raw artifacts.
- Task is ready for QA.

blockers：
- 无

commit：
- 待提交后回填/最终回报提供

提交信息：
- 待提交后回填/最终回报提供
