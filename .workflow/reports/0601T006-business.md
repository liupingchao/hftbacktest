```md
执行线程：
- 业务线程-research

任务ID：
- 0601T006

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/reports/0601T006-business.md`
- `local_live_analysis/cross_exchange_public_sample_xemm_0603_quiet_b/**`
- `local_live_analysis/cross_exchange_public_sample_xemm_0603_quiet_c/**`
- `local_live_analysis/cross_exchange_lead_lag_join_xemm_0603_quiet_b/**`
- `local_live_analysis/cross_exchange_lead_lag_join_xemm_0603_quiet_c/**`
- `local_live_analysis/cross_exchange_lead_lag_analysis_xemm_0603_quiet_b/**`
- `local_live_analysis/cross_exchange_lead_lag_analysis_xemm_0603_quiet_c/**`
- `local_live_analysis/binance_led_hyperliquid_pricing_signal_xemm_0603_quiet_b/**`
- `local_live_analysis/binance_led_hyperliquid_pricing_signal_xemm_0603_quiet_c/**`
- `local_live_analysis/binance_led_hyperliquid_multisample_robustness_0601T006/**`

action：
- 检查 workflow 后确认本地不存在 `.workflow/tasks/0603T006.md` 或 `0603T006` report；当前活跃 T006 任务为 `0601T006`。
- 按用户确认的要求，复用 `xemm_0603_quiet_a` 路径，在 `awsserver1` clean task worktree `/home/admin/hft_live/hftbacktest_0601T006` 再次执行 30min public-only synchronized live data collection。
- 新样本 ID：`xemm_0603_quiet_b`。
- 远端 collection command：
  - `/home/admin/hft_live/venv/bin/python examples/hyperliquid/synchronized_public_collection.py collect --output-dir local_live_analysis/cross_exchange_public_sample_xemm_0603_quiet_b --duration-seconds 1800 --task-id 0601T006 --clean-output`
- 远端 collection 结果：
  - overlap `1800.084405954s`，`passes_min_overlap_600s=true`，`passes_target_1800s=true`。
  - Binance counts：bookTicker `1710706`，depthUpdate `67964`，trade `213073`，depth snapshot present，close reason `duration_elapsed`，reconnect `0`。
  - Hyperliquid counts：l2Book `3330`，trades messages `5833`，trade events `21987`，classification `passes_pricing_research_market_view`，close reason `duration_elapsed`，reconnect `0`。
  - Remote wrapper 自动 Binance alignment returned `1`，按任务规则视为 diagnostic-only，不计入验收路径。
- 已将 `xemm_0603_quiet_b` 从 `awsserver1` 拉回 local machine。
- 本地 raw sha256 校验通过：
  - Binance raw sha256 `87a66fbc57d53edc95bd4822a8a5fc41e74a9df86df34dd043bfe58e7c521552`。
  - Hyperliquid raw sha256 `616324e2c07bf329981cc132ed7f227ea5d2f847f8b26dfdca4cfae14b9d7138`。
- 已将远端 diagnostic alignment 目录移到 local diagnostic-only 路径，并在 local machine 从 copied raw artifacts 重新执行 accepted alignment。
- 本地 Binance alignment：
  - 初次 `--buffer-size 10000000` 失败：`IndexError: event buffer is full; increase --buffer-size`。
  - 使用 `--buffer-size 20000000` rerun 成功。
  - top5 rows `67965`，depth `pu` mismatch `0`，final data row mapping coverage `1.0`，raw message mapping coverage `0.9999507968895601`，first valid update aligned `false`。
- 本地 Hyperliquid alignment：
  - classification `passes_pricing_research_market_view`，synthetic decision count `3600`，topn coverage `1.0`，future/missing joins `0/0`。
- 本地 as-of join / lead-lag / pricing-signal chain：
  - Join rows `3600`，primary usable `3580`，future/missing Binance joins `0/0`。
  - Lead-lag verdict counts：stable `30`，watch `5`，unstable `19`。
  - Pricing signal rows `21445`，primary rows `3580`，single-sample recommendation `keep_for_read_only_research`。
- 已重新运行 aggregate robustness runner，包含：
  - `local_live_analysis/binance_led_hyperliquid_pricing_signal_0601T005`
  - `local_live_analysis/binance_led_hyperliquid_pricing_signal_xemm_0603_quiet_a`
  - `local_live_analysis/binance_led_hyperliquid_pricing_signal_xemm_0603_quiet_b`
- Aggregate result：
  - sample_count `3`
  - recommendation `continue_read_only_runner_refinement`
  - generated outputs include `multi_sample_manifest.json`, `sample_quality_matrix.csv`, `feature_horizon_stability_across_samples.csv`, `effective_horizon_aliasing_by_sample.csv`, `venue_state_conditioning_across_samples.csv`, and `pricing_signal_robustness_recommendation.md`。
- 按用户追加要求，再次在 `awsserver1` 执行 30min public-only synchronized live data collection。
- 新样本 ID：`xemm_0603_quiet_c`。
- 远端 collection command：
  - `/home/admin/hft_live/venv/bin/python examples/hyperliquid/synchronized_public_collection.py collect --output-dir local_live_analysis/cross_exchange_public_sample_xemm_0603_quiet_c --duration-seconds 1800 --task-id 0601T006 --clean-output`
- 远端 collection 结果：
  - overlap `1800.070369182s`，`passes_min_overlap_600s=true`，`passes_target_1800s=true`。
  - Binance counts：bookTicker `1753489`，depthUpdate `68045`，trade `287991`，depth snapshot present。
  - Hyperliquid counts：l2Book `3334`，trades messages `5733`，trade events `29525`，classification `passes_pricing_research_market_view`。
  - Remote wrapper 自动 Binance alignment returned `1`，Hyperliquid alignment returned `0`；remote alignment 按任务规则视为 diagnostic-only，不计入验收路径。
- 已将 `xemm_0603_quiet_c` 从 `awsserver1` 拉回 local machine。
- 本地 raw sha256 校验通过：
  - Binance raw sha256 `51cc0caf5cedc22d7879b09fdca807944a205892ea696587307e6ba614569ab8`。
  - Hyperliquid raw sha256 `6f14e951d85d639dfe561d5a0ac900acb1fd1bbc52d29437f4fc18e0d48d4adb`。
- 已将远端 diagnostic alignment 目录移到 local diagnostic-only 路径，并在 local machine 从 copied raw artifacts 重新执行 accepted alignment。
- 本地 Binance alignment：
  - 使用 `--buffer-size 20000000` 成功。
  - top5 rows `68046`，depth `pu` mismatch `0`，final data row mapping coverage `1.0`，raw message mapping coverage `0.9999241535776283`，first valid update aligned `false`。
- 本地 Hyperliquid alignment：
  - classification `passes_pricing_research_market_view`，synthetic decision count `3599`，topn coverage `1.0`，future/missing joins `0/0`。
- 本地 as-of join / lead-lag / pricing-signal chain：
  - Join rows `3599`，primary usable `3504`，future/missing Binance joins `0/0`。
  - Lead-lag verdict counts：stable `25`，watch `6`，unstable `23`。
  - Pricing signal rows `20990`，primary rows `3504`，single-sample recommendation `keep_for_read_only_research`。
- 已重新运行 aggregate robustness runner，包含：
  - `local_live_analysis/binance_led_hyperliquid_pricing_signal_0601T005`
  - `local_live_analysis/binance_led_hyperliquid_pricing_signal_xemm_0603_quiet_a`
  - `local_live_analysis/binance_led_hyperliquid_pricing_signal_xemm_0603_quiet_b`
  - `local_live_analysis/binance_led_hyperliquid_pricing_signal_xemm_0603_quiet_c`
- Aggregate result：
  - sample_count `4`
  - recommendation `continue_read_only_runner_refinement`
  - sample quality matrix includes all 4 samples with future/missing Binance joins `0/0`。

verify：
- `ssh awsserver1 'ps -p 475728 -o pid,etime,cmd || true'`
- `ssh awsserver1 'cd /home/admin/hft_live/hftbacktest_0601T006 && python3 -m json.tool local_live_analysis/cross_exchange_public_sample_xemm_0603_quiet_b/binance_public_raw/collection_manifest.json >/dev/null && python3 -m json.tool local_live_analysis/cross_exchange_public_sample_xemm_0603_quiet_b/hyperliquid_public_sample/collection_manifest.json >/dev/null'`
- `scp -r awsserver1:/home/admin/hft_live/hftbacktest_0601T006/local_live_analysis/cross_exchange_public_sample_xemm_0603_quiet_b local_live_analysis/`
- `sha256sum local_live_analysis/cross_exchange_public_sample_xemm_0603_quiet_b/binance_public_raw/raw.gz`
- `sha256sum local_live_analysis/cross_exchange_public_sample_xemm_0603_quiet_b/hyperliquid_public_sample/raw.gz`
- `python examples/binance_tick_mm/binance_top5_provenance.py build-sidecars --input-gz local_live_analysis/cross_exchange_public_sample_xemm_0603_quiet_b/binance_public_raw/raw.gz --out-dir local_live_analysis/cross_exchange_public_sample_xemm_0603_quiet_b/binance_alignment --sample-id xemm_0603_quiet_b --symbol BTCUSDT --tick-size 0.1 --opt t --buffer-size 10000000` failed with expected buffer blocker on this larger Binance stream.
- `python examples/binance_tick_mm/binance_top5_provenance.py build-sidecars --input-gz local_live_analysis/cross_exchange_public_sample_xemm_0603_quiet_b/binance_public_raw/raw.gz --out-dir local_live_analysis/cross_exchange_public_sample_xemm_0603_quiet_b/binance_alignment --sample-id xemm_0603_quiet_b --symbol BTCUSDT --tick-size 0.1 --opt t --buffer-size 20000000`
- `python examples/hyperliquid/hyperliquid_raw_alignment.py --input-gzip local_live_analysis/cross_exchange_public_sample_xemm_0603_quiet_b/hyperliquid_public_sample/raw.gz --output-dir local_live_analysis/cross_exchange_public_sample_xemm_0603_quiet_b/hyperliquid_public_sample/alignment --source-label hyperliquid_lag_public_sample_xemm_0603_quiet_b --task-id 0601T006 --collection-manifest local_live_analysis/cross_exchange_public_sample_xemm_0603_quiet_b/hyperliquid_public_sample/collection_manifest.json --recovery-snapshots local_live_analysis/cross_exchange_public_sample_xemm_0603_quiet_b/hyperliquid_public_sample/recovery_snapshots.jsonl --buffer-size 1000000`
- `python examples/hyperliquid/cross_exchange_lead_lag_join.py --sample-dir local_live_analysis/cross_exchange_public_sample_xemm_0603_quiet_b --output-dir local_live_analysis/cross_exchange_lead_lag_join_xemm_0603_quiet_b --binance-symbol BTCUSDT --hyperliquid-coin BTC --tick-size 0.1`
- `python examples/hyperliquid/cross_exchange_lead_lag_analysis.py --input-dir local_live_analysis/cross_exchange_lead_lag_join_xemm_0603_quiet_b --output-dir local_live_analysis/cross_exchange_lead_lag_analysis_xemm_0603_quiet_b`
- `python examples/hyperliquid/binance_led_pricing_signal_runner.py --join-dir local_live_analysis/cross_exchange_lead_lag_join_xemm_0603_quiet_b --analysis-dir local_live_analysis/cross_exchange_lead_lag_analysis_xemm_0603_quiet_b --contract-dir local_live_analysis/binance_led_hyperliquid_data_contract_0601T004 --output-dir local_live_analysis/binance_led_hyperliquid_pricing_signal_xemm_0603_quiet_b`
- `python examples/hyperliquid/binance_led_multi_sample_robustness.py --pricing-signal-dir local_live_analysis/binance_led_hyperliquid_pricing_signal_0601T005 --pricing-signal-dir local_live_analysis/binance_led_hyperliquid_pricing_signal_xemm_0603_quiet_a --pricing-signal-dir local_live_analysis/binance_led_hyperliquid_pricing_signal_xemm_0603_quiet_b --output-dir local_live_analysis/binance_led_hyperliquid_multisample_robustness_0601T006`
- `ssh awsserver1 'cd /home/admin/hft_live/hftbacktest_0601T006 && python3 -m json.tool local_live_analysis/cross_exchange_public_sample_xemm_0603_quiet_c/binance_public_raw/collection_manifest.json >/dev/null && python3 -m json.tool local_live_analysis/cross_exchange_public_sample_xemm_0603_quiet_c/hyperliquid_public_sample/collection_manifest.json >/dev/null'`
- `scp -r awsserver1:/home/admin/hft_live/hftbacktest_0601T006/local_live_analysis/cross_exchange_public_sample_xemm_0603_quiet_c local_live_analysis/`
- `sha256sum local_live_analysis/cross_exchange_public_sample_xemm_0603_quiet_c/binance_public_raw/raw.gz`
- `sha256sum local_live_analysis/cross_exchange_public_sample_xemm_0603_quiet_c/hyperliquid_public_sample/raw.gz`
- `python examples/binance_tick_mm/binance_top5_provenance.py build-sidecars --input-gz local_live_analysis/cross_exchange_public_sample_xemm_0603_quiet_c/binance_public_raw/raw.gz --out-dir local_live_analysis/cross_exchange_public_sample_xemm_0603_quiet_c/binance_alignment --sample-id xemm_0603_quiet_c --symbol BTCUSDT --tick-size 0.1 --opt t --buffer-size 20000000`
- `python examples/hyperliquid/hyperliquid_raw_alignment.py --input-gzip local_live_analysis/cross_exchange_public_sample_xemm_0603_quiet_c/hyperliquid_public_sample/raw.gz --output-dir local_live_analysis/cross_exchange_public_sample_xemm_0603_quiet_c/hyperliquid_public_sample/alignment --source-label hyperliquid_lag_public_sample_xemm_0603_quiet_c --task-id 0601T006 --collection-manifest local_live_analysis/cross_exchange_public_sample_xemm_0603_quiet_c/hyperliquid_public_sample/collection_manifest.json --recovery-snapshots local_live_analysis/cross_exchange_public_sample_xemm_0603_quiet_c/hyperliquid_public_sample/recovery_snapshots.jsonl --buffer-size 1000000`
- `python examples/hyperliquid/cross_exchange_lead_lag_join.py --sample-dir local_live_analysis/cross_exchange_public_sample_xemm_0603_quiet_c --output-dir local_live_analysis/cross_exchange_lead_lag_join_xemm_0603_quiet_c --binance-symbol BTCUSDT --hyperliquid-coin BTC --tick-size 0.1`
- `python examples/hyperliquid/cross_exchange_lead_lag_analysis.py --input-dir local_live_analysis/cross_exchange_lead_lag_join_xemm_0603_quiet_c --output-dir local_live_analysis/cross_exchange_lead_lag_analysis_xemm_0603_quiet_c`
- `python examples/hyperliquid/binance_led_pricing_signal_runner.py --join-dir local_live_analysis/cross_exchange_lead_lag_join_xemm_0603_quiet_c --analysis-dir local_live_analysis/cross_exchange_lead_lag_analysis_xemm_0603_quiet_c --contract-dir local_live_analysis/binance_led_hyperliquid_data_contract_0601T004 --output-dir local_live_analysis/binance_led_hyperliquid_pricing_signal_xemm_0603_quiet_c`
- `python examples/hyperliquid/binance_led_multi_sample_robustness.py --pricing-signal-dir local_live_analysis/binance_led_hyperliquid_pricing_signal_0601T005 --pricing-signal-dir local_live_analysis/binance_led_hyperliquid_pricing_signal_xemm_0603_quiet_a --pricing-signal-dir local_live_analysis/binance_led_hyperliquid_pricing_signal_xemm_0603_quiet_b --pricing-signal-dir local_live_analysis/binance_led_hyperliquid_pricing_signal_xemm_0603_quiet_c --output-dir local_live_analysis/binance_led_hyperliquid_multisample_robustness_0601T006`
- `python -m json.tool local_live_analysis/cross_exchange_public_sample_xemm_0603_quiet_b/run_manifest.json >/dev/null`
- `python -m json.tool local_live_analysis/cross_exchange_public_sample_xemm_0603_quiet_b/sample_manifest.json >/dev/null`
- `python -m json.tool local_live_analysis/cross_exchange_public_sample_xemm_0603_quiet_b/synchronization_quality_summary.json >/dev/null`
- `python -m json.tool local_live_analysis/cross_exchange_public_sample_xemm_0603_quiet_b/binance_alignment/metrics.json >/dev/null`
- `python -m json.tool local_live_analysis/cross_exchange_public_sample_xemm_0603_quiet_b/hyperliquid_public_sample/alignment/metrics.json >/dev/null`
- `python -m json.tool local_live_analysis/cross_exchange_lead_lag_join_xemm_0603_quiet_b/run_manifest.json >/dev/null`
- `python -m json.tool local_live_analysis/cross_exchange_lead_lag_analysis_xemm_0603_quiet_b/run_manifest.json >/dev/null`
- `python -m json.tool local_live_analysis/binance_led_hyperliquid_pricing_signal_xemm_0603_quiet_b/run_manifest.json >/dev/null`
- `python -m json.tool local_live_analysis/binance_led_hyperliquid_multisample_robustness_0601T006/multi_sample_manifest.json >/dev/null`
- `python -m json.tool local_live_analysis/cross_exchange_public_sample_xemm_0603_quiet_c/run_manifest.json >/dev/null`
- `python -m json.tool local_live_analysis/cross_exchange_public_sample_xemm_0603_quiet_c/sample_manifest.json >/dev/null`
- `python -m json.tool local_live_analysis/cross_exchange_public_sample_xemm_0603_quiet_c/synchronization_quality_summary.json >/dev/null`
- `python -m json.tool local_live_analysis/cross_exchange_public_sample_xemm_0603_quiet_c/binance_alignment/metrics.json >/dev/null`
- `python -m json.tool local_live_analysis/cross_exchange_public_sample_xemm_0603_quiet_c/hyperliquid_public_sample/alignment/metrics.json >/dev/null`
- `python -m json.tool local_live_analysis/cross_exchange_lead_lag_join_xemm_0603_quiet_c/run_manifest.json >/dev/null`
- `python -m json.tool local_live_analysis/cross_exchange_lead_lag_analysis_xemm_0603_quiet_c/run_manifest.json >/dev/null`
- `python -m json.tool local_live_analysis/binance_led_hyperliquid_pricing_signal_xemm_0603_quiet_c/run_manifest.json >/dev/null`

done：
- `xemm_0603_quiet_b` 已完成 awsserver1 30min public-only live data collection、local download、local alignment、local as-of join、lead-lag analysis、pricing-signal runner、aggregate robustness rerun。
- `xemm_0603_quiet_c` 已完成 awsserver1 30min public-only live data collection、local download、local alignment、local as-of join、lead-lag analysis、pricing-signal runner、aggregate robustness rerun。
- 本轮满足用户要求的 “just like xemm_0603_quiet_a” repeat collection and local processing。
- 本轮不改变 strategy，不使用 private/order endpoints，不执行 order lifecycle，不做 parameter search，不 default-on/tiny-live/promotion。
- Caveat：`xemm_0603_quiet_a`、`xemm_0603_quiet_b`、`xemm_0603_quiet_c` 都是 quiet-style repeat samples；aggregate 已有 4 个 samples，但仍不等同于覆盖 high-vol / normal-liquidity regime diversity。

blockers：
- 无针对本轮 repeat quiet sample。
- 若按 `0601T006` 原始多样本 regime-diversity 目标继续推进，仍需 high-vol / normal-liquidity windows 或由总控接受当前 quiet-repeat caveat。

commit：
- `2625c05`

提交信息：
- `finalize 0601T006 workflow records`
```
