```md
执行线程：
- 测试线程

任务ID：
- 0526T001

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 用户在执行中明确要求第二次、第三次采集暂时不用做，因此本轮只完成第一次样本 `5-26-active-minmove-control-30min-a` 的采集、拉回、归档和完整 audit/replay/diagnostic 后处理。

files：
- `.workflow/tasks/0526T001.md`
- `.workflow/reports/0526T001-business.md`
- `task_plan.md`
- `progress.md`
- `findings.md`
- `.workflow/dashboard.html`
- `.workflow/dispatch_suggestions.md`
- `local_live_analysis/5-26-active-minmove-control-30min-a/**`
- `local_live_analysis/archive/5-26-active-minmove-control-30min-a.tar.gz`
- `local_live_analysis/archive/5-26-active-minmove-control-30min-a.tar.gz.sha256`
- `local_live_analysis/stage9d_candidate_bucket_refinement_0526T001_aggregate/**`

action：
- `0525T001` QA 已通过后，正式执行 `0526T001`。
- 使用 remote clean worktree `/home/admin/hft_live/worktrees/0526T001-active-minmove` 和 commit `a1ee78b` 采集 `5-26-active-minmove-control-30min-a`。
- 采集保持 current-format no-rule / default-off，未启用 candidate，未放宽 guard，未修改策略行为。
- 远端 run dir 原始 raw 路径为 `raw_market_data/`；为了兼容 `align_live_run.py` 的既有 fetch 合约，远端补了 task-scoped `data -> raw_market_data` symlink。
- 本地拉回后完成 normal replay、audit replay、maker acceptance、T009 sidecar/join、Stage 5 labels、Step 5C diagnostics、Stage 6 calibration、Step 9B replay、Step 9D 单样本分类，以及 5+1 样本 aggregate Step 9D rerun。
- 为满足 Step 9B 既有输入合约，给新样本补了 `stage8b_quote_update_diagnostic_0519T005/implementation_planning_decision.json` derived placeholder。该文件只声明 default-off replay input available，不代表新的 Step 8B 结论或策略授权。
- 用户要求第二次、第三次采集暂缓后，未启动 `5-26-active-minmove-control-30min-b/c`。

collection：
- run id: `5-26-active-minmove-control-30min-a`
- deployed commit: `a1ee78b`
- preflight: passed, `dirty=false`, `compatibility.passed=true`, `audit_field_count=159`
- stopper:
  - armed: `2026-05-26T01:29:22+09:00`
  - stop sequence began: `2026-05-26T01:59:22+09:00`
  - finished: `2026-05-26T02:00:12+09:00`
- stop marker exit code: `0`
- live audit rows / fields: `64426 / 159`
- T006 missing fields: `[]`
- raw gzip: `local_live_analysis/5-26-active-minmove-control-30min-a/raw_market_data/btcusdt_20260525.gz`
- raw gzip check: passed

alignment / acceptance：
- audit replay:
  - common rows `48415`
  - action match rate `1.0`
  - planned action match rate `1.0`
  - reject reason match rate `1.0`
  - throttle reason match rate `1.0`
  - strict replay lag gate passed `true`
  - breaches/drops/failures `0/0/0`
- maker acceptance:
  - `maker_acceptance.json`: passed `true`, hard failures `[]`
  - `maker_acceptance_with_market_view.json`: passed `true`, hard failures `[]`
- market-view sidecar/join:
  - first valid update aligned `true`
  - depth `pu` mismatch `0`
  - decision join coverage `1.0`
  - future join count `0`
  - gap-crossed join count `0`
  - join missing count `0`
  - stale join count `258`
  - top5 join age p99 `27.2403406ms`
  - top5 tick match `0.9846948259836827`
  - top5 qty match `0.9711039966952391`

Stage 5：
- submit orders `2526`
- filled orders `46`
- fill-after-cancel orders `30`
- fill by horizon:
  - `100ms`: `3`
  - `500ms`: `18`
  - `1000ms`: `25`
  - `5000ms`: `40`

Step 5C：
- decision rows `48416`
- bookTicker anchor rows `40611`
- guarded depth fallback rows `7805`
- missing anchor rows `0`
- stale anchor rows `0`
- bid/ask clamped rows `1230 / 1976`
- post-only risk after recheck rows `0`

Stage 6：
- decision state `requires_more_current_format_samples`
- live/replay submit orders `2526 / 2526`
- matched submit orders `2526`
- live/replay filled orders `46 / 46`
- live/replay fill-after-cancel orders `30 / 30`

Step 9B：
- classification `promising_but_single_sample`
- decision rows `48416`
- submit orders `2526`
- missing T006 field count `0`
- maker acceptance passed `true`
- no live/default-on/promotion/sample-expansion/production behavior change authorized.

Step 9D：
- 单样本 fine-bucket result：
  - `min_move_quote_age_churn_guard`: stable `0`, needs-more-fills `27`, reject `0`
  - `inventory_reservation_shift_band`: stable `0`, needs-more-fills `13`, reject `12`
  - `size_reduction_or_add_side_suppression_pressure`: stable `0`, needs-more-fills `10`, reject `14`
- 解释：单样本不用于稳定性 promotion；该样本的价值是增加 clean active fills 和目标 bucket 覆盖。
- 5+1 aggregate fine-bucket rerun：
  - `min_move_quote_age_churn_guard`: stable promising `20`, parameter-sweep seed buckets `5`, reject `0`
  - aggregate seed buckets: `api_churn=young_quote_churn`, `inventory_state=large_skew_or_low_score`, `latency_stale_age=stale_latency_medium`, `volatility_markout_dispersion=volatility_high`, `volatility_markout_dispersion=volatility_medium`
  - `inventory_reservation_shift_band`: seed `0`, reject `19`
  - `size_reduction_or_add_side_suppression_pressure`: seed `1`, reject `19`, remaining seed is `volatility_markout_dispersion=volatility_high`
- 新样本贡献 `46` fills，使 global filled-order gap 从 `160` 缩小到约 `114`。仍不满足 `ready_for_tiny_live_design`。

archive：
- archive: `local_live_analysis/archive/5-26-active-minmove-control-30min-a.tar.gz`
- sha256: `2add6958e266cffd2911b70349ddf45001b06314ea414446401d0be468bd4a7f`
- archive gzip check: passed
- archive includes deployment manifest, start marker, stop marker, raw gzip, audit, logs, replay outputs, sidecar/join, Stage 5/5C/6/9 outputs, file manifest and SHA256SUMS.

verify：
- `python -m pytest examples/binance_tick_mm/test_deploy_preflight.py` -> passed, `5 passed`
- `python -m pytest examples/binance_tick_mm/test_quote_adjustment_replay.py` -> passed, `5 passed`
- `python examples/binance_tick_mm/deploy/preflight_live_run.py --help` -> passed
- `python examples/binance_tick_mm/align_live_run.py --help` -> passed
- `python examples/binance_tick_mm/maker_acceptance.py --help` -> passed
- `python examples/binance_tick_mm/quote_adjustment_replay.py --help` -> passed
- `python examples/binance_tick_mm/candidate_bucket_refinement.py --help` -> passed
- `bash -n examples/binance_tick_mm/deploy/run_live.sh` -> passed
- `python examples/binance_tick_mm/align_live_run.py --run-id 5-26-active-minmove-control-30min-a --local-root local_live_analysis --remote-host admin@awsserver1 --remote-root /home/admin/hft_live` -> passed after adding remote task-scoped `data -> raw_market_data` symlink
- `gzip -t local_live_analysis/5-26-active-minmove-control-30min-a/raw_market_data/*.gz` -> passed
- T009 sidecar build and decision join -> passed
- Stage 5 labels -> passed
- Step 5C diagnostics -> passed
- Stage 6 calibration -> passed
- Step 9B runner -> passed
- Step 9D single-sample and aggregate rerun -> passed
- `gzip -t local_live_analysis/archive/5-26-active-minmove-control-30min-a.tar.gz` -> passed

done：
- 本轮完成第一次 targeted active current-format no-rule/default-off sample 的完整采集和后处理。
- 第二次、第三次采集按用户最新指令暂缓，未启动。
- 结果支持继续围绕 `min_move_quote_age_churn_guard` 做 multi-sample parameter sweep design，但仍需要更多 natural fills；不支持 tiny live，不支持 default-on，不支持 guard relaxation。

blockers：
- 无执行 blocker。
- 剩余 gap：本轮只新增 `46` fills，距离 `500` fill tiny-live mass 仍约缺 `114` fills。

commit：
- 待总控收尾提交

提交信息：
- 待总控收尾提交
```
