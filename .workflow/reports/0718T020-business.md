# 线程回报

执行线程：
- 总控 auto-loop / 业务实现线程

任务ID：
- 0718T020

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0718T020.md`
- `examples/hyperliquid/cross_exchange_online_estimators.py`
- `examples/hyperliquid/hyperliquid_tiny_live_m2_public_watcher.py`
- `examples/hyperliquid/test_cross_exchange_fill_feedback.py`
- `local_live_analysis/principal_alignment_task9_0718T020/`

action：
- 新增 T018/T019 artifact 兼容的 quote lifecycle normalizer，使用 `attempt_key` 和 ledger `fill_id` 做幂等身份。
- rejected/never-resting 排除；short hold、run-end、forced cancel、missing terminal public coverage 和身份冲突进入 censored/fail-closed，不计为普通 no-fill。
- partial fill 保留 `filled/original` ratio；统计使用原始数量和 resting exposure seconds 的 pooled exposure-weighted fill ratio，并输出 public arrival 统计。
- 新增带 target/min-observation/min-exposure、hysteresis、rate-limit、anti-windup、bounds、version 的 observe-only feedback candidate；candidate 不接入报价、下单、撤单或 activation。
- 新增 checksum/schema 校验的 controller state restore；失败回到 neutral。
- watcher 的 public shadow、event-driven、inline 收尾路径均写出 lifecycle、aggregate、quarantine、candidate、state、snapshot 和 manifest，并在 `live_status.json` 暴露 feedback snapshot。

verify：
- `python -m pytest examples/hyperliquid/test_cross_exchange_fill_feedback.py -q`：`5 passed`。
- `python -m pytest examples/hyperliquid/test_cross_exchange_online_estimators.py -q`：`8 passed`。
- `python -m pytest examples/hyperliquid -q`：`446 passed`。
- modified Python `py_compile`、CLI `--help`、`git diff --check` 通过。
- 远端隔离 clone `/home/admin/hftbacktest-cross-exchange-0718T020` 使用既有 Hyperliquid SDK venv 执行第二次 60 秒 public-only run：
  - `60.000712s`；
  - `113` L2、`85` trade、`148` evaluations、`0` reconnect；
  - `0` quote exposure lifecycle、`0` feedback observations、candidate `unavailable_neutral`；
  - credentials/private/account/order/cancel 全 false；
  - `dynamic_spread_activation_enabled=false`、`fill_feedback_activation_enabled=false`、`actual_quote_behavior_changed=false`。
- 第一次 `/usr/bin/python3` public-only attempt 因远端缺少 websocket package 在 `1.500913s` fail-closed，未调用任何 private/order endpoint；该环境阻塞也被保留为独立 artifact。
- 第二次 remote/local artifact 各 `29` 个文件，逐文件 SHA-256 完全一致。
- T020 feedback replay：`lifecycle_row_count=0`，source/replay snapshot SHA-256 exact match。
- T019 estimator replay：`198` event rows、`0` exposure rows，source/replay snapshot SHA-256 exact match。

done：
- 完成 Principal Alignment Task 9 的 lifecycle、censoring、partial/dedup、exposure-weighted aggregation、bounded observe-only feedback、restart checksum 和 replay contract。
- 真实 live 中没有 resting lifecycle 的事实被如实保留；没有把 public-only exposure 或 fallback candidate 当作 fill calibration。
- 实际 quote/order behavior 未改变。

blockers：
- 无实现阻塞。
- 真实 resting/fill lifecycle 仍未获得，因此 feedback candidate 保持 neutral/unavailable；dynamic spread、fill feedback activation、multi-level 和 promotion 不被本任务授权。

commit：
- `f04333d`

提交信息：
- `Add observe-only exposure-weighted fill feedback`
