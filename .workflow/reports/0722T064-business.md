# 0722T064 Business Report

执行线程：
- 总控 / 业务执行线程

任务ID：
- 0722T064

状态：
- 待验收

更新时间：
- 2026-07-22 17:00 Asia/Shanghai

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `examples/hyperliquid/cross_exchange_adaptive_live_reachability_preflight.py`
- `examples/hyperliquid/test_cross_exchange_adaptive_live_reachability_preflight.py`
- `local_live_analysis/adaptive_live_evidence_reachability_0722T064/`
- `.workflow/tasks/0722T064.md`
- workflow tracking files

action：
- 通过 watcher source SHA 和 token line-order 冻结 exact timeline：
  manager call line `10985`；dynamic candidate argument `11004`；
  fill-feedback candidate argument `11016`；dynamic exposure observation
  `11069`；fill-feedback finalization `11595`。
- SSM read-only 审计 T047/T052 既有 redacted estimator/lifecycle artifacts，
  记录 command IDs、remote roots 和 10 个 file SHA-256。
- T047 dynamic：
  buy/sell 各 `4` observations，但各自只有一个 distance，
  双侧均 `insufficient_distance_variation`。
- T052 dynamic：
  buy `0` observations；sell `4` observations/`3` distances，只有 sell fit
  pass。
- 合并 T047+T052：
  buy `4` observations/`1` distance；sell `8` observations/`4` distances；
  buy side 仍不满足 dynamic seed。
- T047/T052 fill feedback：
  合计 `4` lifecycle rows，但 `0` eligible、`0s` eligible exposure；
  rejected 被排除，resting rows 因 interval bounds 缺失被 censor。
- Current live profile 只有一个 manager cycle；candidate 在 cycle 前读取，
  当前 cycle 产生的 evidence 无法影响已提交 quote。

verify：
- Focused：
  `3 passed in 0.03s`。
- Full Hyperliquid：
  `1286 passed, 2 skipped in 57.48s`。
- Official artifact deterministic rerun：
  全部 SHA-256 一致。
- Conda `py_compile`、absolute-path scan、`git diff --check`：通过。
- AWS instance/status/SSM read-only preflight：
  instance/system `ok`，SSM `Online`，`xemm.service inactive/disabled`，
  无 watcher/orchestrator process。
- 本任务未读取 credentials，未调用 private/account/order/cancel，未启动
  service/watcher/orchestrator，未产生订单。

done：
- Current single-cycle adaptive activation 被证明不可达，不再浪费 live
  window 重复 fallback。
- T047/T052 被证明不具备 dynamic 或 fill-feedback prewarm 资格。
- Final recommendation：
  `route_to_public_multi_distance_dynamic_seed_then_three_window_dynamic_live`。
- 冻结后续顺序：
  1. public-only multi-distance dynamic calibration/seed contract；
  2. fresh-authorized three-window dynamic live；
  3. fill-feedback target ratification；
  4. fresh-authorized fill-feedback active live。

blockers：
- 独立 QA 验收。
- 后续 live step 2/4 仍需新的 exact envelope 授权。

commit：
- 待提交

提交信息：
- `Prove adaptive live evidence reachability`

