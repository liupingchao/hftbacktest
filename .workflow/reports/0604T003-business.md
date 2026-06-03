```md
执行线程：
- 业务线程-research

任务ID：
- 0604T003

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0604T003.md`
- `.workflow/reports/0604T003-business.md`
- `examples/hyperliquid/binance_led_multi_sample_robustness.py`
- `examples/hyperliquid/test_binance_led_multi_sample_robustness.py`
- `local_live_analysis/event_mode_canonical_pricing_signal_0604T003/**`
- `findings.md`
- `progress.md`

action：
- 将 Binance-led Hyperliquid multi-sample robustness 的正式判断口径改为 canonical event-mode：
  - 从 pricing-signal manifest 追溯 `join_sample_manifest -> source_sample_dir -> hyperliquid_public_sample/alignment/run_manifest.json/metrics.json`。
  - 识别 `decision_mode=event` 为 `canonical_event_mode`。
  - 识别缺省/旧 fixed-grid synthetic artifacts 为 `diagnostic_only_synthetic_decision_grid`。
- 在 `sample_quality_matrix.csv` / manifest / stability rows / aliasing rows 中新增或输出：
  - `decision_mode`
  - `canonical_status`
  - `alignment_run_manifest`
  - `alignment_metrics`
  - `event_decision_count`
  - `synthetic_decision_count`
  - `independent_future_row_delta_count`
  - `horizon_future_row_delta_groups`
  - `canonical_eligible_sample_count`
  - `canonical_independent_future_row_delta_count`
- Recommendation logic 现在只使用 canonical event-mode samples；synthetic fixed-grid samples 可解析、可输出诊断表，但不能驱动 canonical recommendation。
- Venue-state conditioning aggregate 只使用 canonical samples。
- 保持 read-only/offline 边界；没有重新采集数据，没有接 private/order endpoints，没有执行 order lifecycle，没有修改 strategy，没有 parameter search / default-on / tiny-live / promotion。

task-scoped outputs：
- Canonical event-mode aggregate：
  - output: `local_live_analysis/event_mode_canonical_pricing_signal_0604T003/`
  - input samples: `xemm_0603_quiet_a_event`, `xemm_0603_quiet_b_event`, `xemm_0603_quiet_c_event`
  - `sample_count=3`
  - `canonical_sample_count=3`
  - `diagnostic_synthetic_sample_count=0`
  - recommendation: `continue_read_only_runner_refinement`
  - reason: `all allowlist features have stable multi-sample core evidence`
- Synthetic diagnostic comparison：
  - output: `local_live_analysis/event_mode_canonical_pricing_signal_0604T003/synthetic_diagnostic_comparison/`
  - input samples: ordinary synthetic `xemm_0603_quiet_a/b/c`
  - `sample_count=3`
  - `canonical_sample_count=0`
  - `diagnostic_synthetic_sample_count=3`
  - recommendation: `needs_more_public_samples`
  - reason: `fewer than three canonical event-mode synchronized public samples are available`

synthetic downgrade behavior：
- Ordinary synthetic a/b/c artifacts still parse and generate diagnostic outputs.
- They are explicitly marked `diagnostic_only_synthetic_decision_grid`.
- They no longer count toward `canonical_sample_count`.
- Their mixed or stable directions do not drive `continue_read_only_runner_refinement` or `reject_for_runner_design`.

event-mode canonical behavior：
- Event-mode a/b/c artifacts are marked `canonical_event_mode`.
- The aggregate records independent future-row-delta groups by horizon, for example `100:1,2`, `500:1,2,3`, and longer horizon groups.
- Final recommendation uses canonical samples and independent future-row-delta diagnostics rather than nominal horizon count alone.

verify：
- `python examples/hyperliquid/binance_led_pricing_signal_runner.py --help` passed.
- `python examples/hyperliquid/binance_led_multi_sample_robustness.py --help` passed.
- `python -m py_compile examples/hyperliquid/binance_led_pricing_signal_runner.py examples/hyperliquid/binance_led_multi_sample_robustness.py` passed.
- `python -m pytest examples/hyperliquid/test_binance_led_pricing_signal_runner.py examples/hyperliquid/test_binance_led_multi_sample_robustness.py` passed: `7 passed`.
- Task-scoped canonical event-mode aggregate passed:
  - `python examples/hyperliquid/binance_led_multi_sample_robustness.py --pricing-signal-dir local_live_analysis/event_horizon_comparison_0604T002/binance_led_hyperliquid_pricing_signal_xemm_0603_quiet_a_event --pricing-signal-dir local_live_analysis/event_horizon_comparison_0604T002/binance_led_hyperliquid_pricing_signal_xemm_0603_quiet_b_event --pricing-signal-dir local_live_analysis/event_horizon_comparison_0604T002/binance_led_hyperliquid_pricing_signal_xemm_0603_quiet_c_event --output-dir local_live_analysis/event_mode_canonical_pricing_signal_0604T003`
- Synthetic diagnostic comparison passed:
  - `python examples/hyperliquid/binance_led_multi_sample_robustness.py --pricing-signal-dir local_live_analysis/binance_led_hyperliquid_pricing_signal_xemm_0603_quiet_a --pricing-signal-dir local_live_analysis/binance_led_hyperliquid_pricing_signal_xemm_0603_quiet_b --pricing-signal-dir local_live_analysis/binance_led_hyperliquid_pricing_signal_xemm_0603_quiet_c --output-dir local_live_analysis/event_mode_canonical_pricing_signal_0604T003/synthetic_diagnostic_comparison`
- Generated manifests parsed with `python -m json.tool`:
  - `local_live_analysis/event_mode_canonical_pricing_signal_0604T003/multi_sample_manifest.json`
  - `local_live_analysis/event_mode_canonical_pricing_signal_0604T003/synthetic_diagnostic_comparison/multi_sample_manifest.json`
- `git diff --check` passed.

done：
- Canonical event-mode aggregate path is now implemented for multi-sample robustness.
- Ordinary synthetic fixed-grid artifacts are downgraded to diagnostic-only and no longer count as canonical short-horizon independent evidence.
- Task-scoped event-mode and synthetic comparison artifacts were generated locally from existing inputs only.

blockers：
- 无。

commit：
- a95201e

提交信息：
- Canonicalize event-mode pricing robustness
```
