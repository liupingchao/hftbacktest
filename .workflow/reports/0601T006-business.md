```md
执行线程：
- 业务线程-research

任务ID：
- 0601T006

状态：
- 执行中

是否进行QA验收：
- 否

QA说明：
- 当前任务结果暂不进入QA验收，待 public-only 多样本采集、逐样本 join/analyze/pricing-signal rerun、aggregate robustness 输出完成后再派发 QA。

files：
- `.workflow/tasks/0601T006.md`
- `.workflow/reports/0601T006-business.md`
- `examples/hyperliquid/binance_led_multi_sample_robustness.py`
- `examples/hyperliquid/test_binance_led_multi_sample_robustness.py`
- `local_live_analysis/binance_led_hyperliquid_multisample_robustness_0601T006/**`
- `progress.md`

action：
- 完成采集前准备，不启动网络采集。
- 新增 offline/read-only multi-sample aggregate runner：`examples/hyperliquid/binance_led_multi_sample_robustness.py`。
- 新增 focused tests：`examples/hyperliquid/test_binance_led_multi_sample_robustness.py`。
- 使用现有 `0601T005` 单样本 pricing-signal artifacts 生成 preparation baseline aggregate outputs：
  - `multi_sample_manifest.json`
  - `sample_quality_matrix.csv`
  - `feature_horizon_stability_across_samples.csv`
  - `effective_horizon_aliasing_by_sample.csv`
  - `venue_state_conditioning_across_samples.csv`
  - `pricing_signal_robustness_recommendation.md`
- 当前 baseline recommendation 为 `needs_more_public_samples`，原因是当前仅处理 `1` 个 synchronized public sample，符合任务预期。
- 已确认采集 wrapper 支持后续使用 `collect --output-dir --duration-seconds --task-id` 启动 public-only synchronized collection。

verify：
- `python examples/hyperliquid/binance_led_multi_sample_robustness.py --help`
- `python -m py_compile examples/hyperliquid/binance_led_multi_sample_robustness.py examples/hyperliquid/test_binance_led_multi_sample_robustness.py`
- `python -m pytest examples/hyperliquid/test_binance_led_multi_sample_robustness.py -q`
- `python examples/hyperliquid/binance_led_multi_sample_robustness.py --output-dir local_live_analysis/binance_led_hyperliquid_multisample_robustness_0601T006`
- `python -m json.tool local_live_analysis/binance_led_hyperliquid_multisample_robustness_0601T006/multi_sample_manifest.json`

done：
- 采集前准备完成。
- 后续仍需用户选择/批准实际 high-vol / quiet / normal public-only collection windows。
- 后续仍需为每个新样本运行 synchronized collection、as-of join、lead-lag analysis、pricing-signal runner，然后重新运行 aggregate robustness runner。

blockers：
- 等待高波动/低波动/正常流动性窗口选择与实际 public-only 样本采集。

commit：
- 待提交

提交信息：
- 待提交
```
