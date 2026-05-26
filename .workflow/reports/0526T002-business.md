```md
执行线程：
- 测试线程

任务ID：
- 0526T002

状态：
- 待验收

是否进行QA验收：
- 是

files：
- `.workflow/tasks/0526T002.md`
- `.workflow/reports/0526T002-business.md`
- `task_plan.md`
- `progress.md`
- `findings.md`
- `.workflow/dashboard.html`
- `.workflow/dispatch_suggestions.md`
- `local_live_analysis/5-26-active-minmove-control-60min-a/**`
- `local_live_analysis/archive/5-26-active-minmove-control-60min-a.tar.gz`
- `local_live_analysis/archive/5-26-active-minmove-control-60min-a.tar.gz.sha256`
- `local_live_analysis/stage9d_candidate_bucket_refinement_0526T002_aggregate/**`

action：
- `0526T001` QA 已通过后，正式执行 `0526T002`。
- 使用 remote clean worktree `/home/admin/hft_live/worktrees/0526T002-active-minmove-60min` 和 commit `76130f4` 采集 `5-26-active-minmove-control-60min-a`。
- baseline config 来自已验收样本 `5-26-active-minmove-control-30min-a`；diff 只包含 `output_root`、`run_id_prefix`、`audit_csv`、`live_safety.connector_config` 的 run-path / run-id 替换。未修改 quote/risk/guard/strategy 参数。
- 采集保持 current-format no-rule / default-off，未启用 candidate，未放宽 guard，未修改策略行为，未做 tiny live 或 promotion。
- 远端 run dir 补了 task-scoped `data -> raw_market_data` symlink 以兼容 `align_live_run.py` 既有 fetch 合约；没有移动或删除 raw 文件。
- 本地拉回后完成 normal replay、audit replay、maker acceptance、T009 sidecar/join、Stage 5 labels、Step 5C diagnostics、Stage 6 calibration、Step 9B replay、Step 9D 单样本分类，以及 5+1+1 aggregate Step 9D rerun。
- 为满足 Step 9B 既有输入合约，给新样本补了 `stage8b_quote_update_diagnostic_0519T005/implementation_planning_decision.json` derived placeholder。该文件只声明 default-off replay input available，不代表新的 Step 8B 结论或策略授权。

collection：
- run id: `5-26-active-minmove-control-60min-a`
- deployed commit: `76130f4`
- preflight: passed, `dirty=false`, `compatibility.passed=true`, `audit_field_count=159`
- stopper:
  - start marker: `2026-05-26T01:35:44Z`
  - stop marker: `2026-05-26T02:35:44Z`
  - stop marker exit code: `0`
- live audit rows / fields: `150756 / 159`
- T006 missing fields: `[]`
- raw gzip: `local_live_analysis/5-26-active-minmove-control-60min-a/raw_market_data/btcusdt_20260526.gz`
- raw gzip check: passed
- remote live status after stop: no remaining live process; only the status-check command itself matched `pgrep`.

alignment / acceptance：
- audit replay:
  - common rows `118508`
  - action match rate `1.0`
  - planned action match rate `1.0`
  - reject reason match rate `1.0`
  - throttle reason match rate `1.0`
  - strict replay lag gate passed `true`
  - breaches/drops/failures `0/0/0`
  - consumed/scheduled `118508 / 118509`
- base maker acceptance:
  - `maker_acceptance.json`: passed `true`, hard failures `[]`
- market-view maker acceptance:
  - `maker_acceptance_with_market_view.json`: passed `false`
  - classification: `limited_pricing_research`
  - only hard failure: `top5_join_age_ms_p99 = 69.99774251999995ms`, expected `<= 50ms`
- market-view sidecar/join:
  - first valid update aligned `true`
  - depth `pu` mismatch `0`
  - decision join coverage `1.0`
  - future join count `0`
  - gap-crossed join count `0`
  - join missing count `0`
  - stale join count `1809`
  - stale join rate `0.015264663443282788`
  - bookTicker/depth BBO mismatch rate `2.2243642025654334e-05`
  - top5 tick match `0.8929608127721336`
  - top5 qty match `0.8486093765821716`
- interpretation:
  - 该样本结构上是干净的：future/gap/missing join 均为 `0`，top5 tick/qty 和 bookTicker/depth BBO 检查达标。
  - 但它不是 strict-clean market-view sample，因为 top5 join age p99 超过 50ms；后续 aggregate 中应作为 caveated / limited-pricing-research 输入处理。

Stage 5：
- submit orders `5095`
- filled orders `164`
- fill-after-cancel orders `81`
- fill by horizon:
  - `100ms`: `22`
  - `500ms`: `73`
  - `1000ms`: `92`
  - `5000ms`: `137`
- observed-only proxy 边界保持不变：queue / priority、missed opportunity、realized PnL decomposition 不被解释为 exact queue 或 counterfactual proof。

Step 5C：
- decision rows `118509`
- bookTicker anchor rows `89099`
- guarded depth fallback rows `28679`
- missing anchor rows `0`
- stale anchor rows `731`
- bid/ask clamped rows `2436 / 5537`
- suppress buy/sell rows `731 / 731`
- post-only risk after recheck rows `0`

Stage 6：
- decision state `methodology_valid_single_sample`
- live/replay submit orders `5095 / 5095`
- matched submit orders `5095`
- live/replay filled orders `164 / 163`
- live/replay fill-after-cancel orders `81 / 80`
- aligned-enough checks passed; no not-aligned rows were reported in the Stage 6 summary.

Step 9B：
- classification `promising_but_single_sample`
- decision rows `118509`
- submit orders `5095`
- missing T006 field count `0`
- maker acceptance passed `true`
- no live/default-on/promotion/sample-expansion/production behavior change authorized.

Step 9D：
- 单样本 fine-bucket result：
  - `min_move_quote_age_churn_guard`: stable `0`, needs-more-fills `26`, reject `1`
  - `inventory_reservation_shift_band`: stable `0`, needs-more-fills `19`, reject `6`
  - `size_reduction_or_add_side_suppression_pressure`: stable `0`, needs-more-fills `19`, reject `5`
- 解释：单个 1H 样本提供了更多 fills，但仍不能独立给出 stable bucket 或 promotion 结论。
- 5+1+1 aggregate fine-bucket rerun：
  - caveated samples: `5-19-night-active-30min-a`, `5-26-active-minmove-control-60min-a`
  - `min_move_quote_age_churn_guard`: stable promising `19`, parameter-sweep seed buckets `2`, reject `1`
  - aggregate seed buckets: `inventory_state=large_skew_or_low_score`, `latency_stale_age=stale_latency_medium`
  - diagnostic-only stable buckets: `adverse_5s_markout`, `fill_after_cancel`, `positive_spread_capture`, `strict_clean_market_view`, `trade_intensity_high`, `trade_intensity_medium`
  - `inventory_reservation_shift_band`: stable `2`, seed `0`, reject `19`
  - `size_reduction_or_add_side_suppression_pressure`: stable `1`, seed `0`, reject `20`
- 解释：
  - `min_move_quote_age_churn_guard` 仍是最合理的后续 parameter-sweep 主线，但新 1H 样本把 seed 从更宽泛的 5 个缩窄为 2 个 decision-time-visible bucket。
  - `inventory_reservation_shift_band` 和 `size_reduction_or_add_side_suppression_pressure` 不适合成为近期开主线的 parameter sweep。

fill mass / tiny-live readiness：
- 该 1H 样本新增 `164` natural fills，明显高于 T001 30min 样本的 `46` fills。
- 如果把 caveated 1H 样本计入总量，filled-order mass 从 `386` 增加到约 `550`，数字上超过 Step 9C `500` fill 门槛。
- 但由于该 1H 样本 market-view strict gate 未通过，strict clean-only fill mass 仍约为 `386 / 500`，仍缺约 `114` fills。
- 因此本轮减少了 filled-order mass 风险，并给参数搜索方向提供更多证据，但仍不满足 `ready_for_tiny_live_design`。

archive：
- archive: `local_live_analysis/archive/5-26-active-minmove-control-60min-a.tar.gz`
- sha256: `e5e4fcbc47c17292937d16a1b15429e1f5b81b46e8915bd669d92b40aae28403`
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
- `python examples/binance_tick_mm/align_live_run.py --run-id 5-26-active-minmove-control-60min-a --local-root local_live_analysis --remote-host admin@awsserver1 --remote-root /home/admin/hft_live` -> passed after adding remote task-scoped `data -> raw_market_data` symlink
- `gzip -t local_live_analysis/5-26-active-minmove-control-60min-a/raw_market_data/*.gz` -> passed
- T009 sidecar build and decision join -> passed
- Stage 5 labels -> passed
- Step 5C diagnostics -> passed
- Stage 6 calibration -> passed
- Step 9B runner -> passed
- Step 9D single-sample and aggregate rerun -> passed
- `gzip -t local_live_analysis/archive/5-26-active-minmove-control-60min-a.tar.gz` -> passed

done：
- 本轮完成一个 targeted active current-format no-rule/default-off 1H sample 的完整采集和后处理。
- 结果支持继续围绕 `min_move_quote_age_churn_guard` 做 parameter-sweep design，尤其是 `large_skew_or_low_score` 和 `stale_latency_medium` 两个 seed bucket。
- 结果不支持 tiny live，不支持 default-on，不支持 guard relaxation。

blockers：
- 无执行 blocker。
- QA 需要决定是否接受该样本作为 caveated evidence；它不应被计为 strict clean market-view fill mass。

commit：
- 待总控收尾提交

提交信息：
- 待总控收尾提交
```
