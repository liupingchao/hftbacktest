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
- `task_plan.md`

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
- 用户确认本任务真实采集环境应为 `awsserver1`，不是 local machine，避免不同机器看到的数据差异。
- 已将任务边界更新为：fresh public synchronized collection 必须在 `awsserver1` 的 clean task worktree 执行；不得覆盖或 reset 现有 dirty live worktree。
- 用户进一步明确：`awsserver1` 只作为 public raw collection / livetest 机器；accepted alignment、as-of join、lead-lag analysis、pricing-signal runner、aggregate robustness 必须在 local machine 执行。
- 任务边界已更新为：从 `awsserver1` 拉回 public raw artifacts/manifests 后，在 local machine 重新 alignment 和后续分析；远端 alignment 如被 full wrapper 自动生成，仅作为 diagnostic-only，不计入验收。
- 本地曾尝试 `xemm_0603_quiet_a` 采集，但该尝试无效：
  - 执行机器错误：local machine，不是 `awsserver1`。
  - 网络质量失败：Binance 和 Hyperliquid 均为 `[Errno -3] Temporary failure in name resolution`。
  - overlap 仅 `1.503s`，未达到 `600s` 最低门槛和 `1800s` 目标。
  - 该目录不得计入 `0601T006` accepted samples。
- 已将代码同步到 `awsserver1` clean worktree：
  - `/home/admin/hft_live/hftbacktest_0601T006`
  - HEAD `3496ae6`
  - 未修改现有 dirty live worktree `/home/admin/hft_live/hftbacktest`
- `awsserver1` clean worktree 使用 `/home/admin/hft_live/venv/bin/python`；已安装 `websocket-client==1.9.0` 以支持 public WebSocket collection。
- 已在 `awsserver1` 启动 `xemm_0603_quiet_a` 远端 public raw collection，requested duration `1800s`。当前采集仍在进行；完成后需拉回 raw artifacts 并在 local machine 重新 alignment。

verify：
- `python examples/hyperliquid/binance_led_multi_sample_robustness.py --help`
- `python -m py_compile examples/hyperliquid/binance_led_multi_sample_robustness.py examples/hyperliquid/test_binance_led_multi_sample_robustness.py`
- `python -m pytest examples/hyperliquid/test_binance_led_multi_sample_robustness.py -q`
- `python examples/hyperliquid/binance_led_multi_sample_robustness.py --output-dir local_live_analysis/binance_led_hyperliquid_multisample_robustness_0601T006`
- `python -m json.tool local_live_analysis/binance_led_hyperliquid_multisample_robustness_0601T006/multi_sample_manifest.json`

done：
- 采集前准备完成。
- 已通过 git 将当前代码同步到 `awsserver1` 的 clean task worktree。
- 后续在 `awsserver1` 执行 3 个 `1800s` public-only raw samples：
  - `xemm_<MMDD>_highvol_a`
  - `xemm_<MMDD>_quiet_a`
  - `xemm_<MMDD>_normal_a`
- 后续需把每个样本的 public raw artifacts/manifests 拉回 local machine，再在 local machine 运行 alignment、as-of join、lead-lag analysis、pricing-signal runner，然后重新运行 aggregate robustness runner。

blockers：
- 等待 `xemm_0603_quiet_a` 远端 raw collection 完成并拉回 local machine 做本地 alignment；后续还需 high-vol / normal windows。

commit：
- 待提交

提交信息：
- 待提交
```
