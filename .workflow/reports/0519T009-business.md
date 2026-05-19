```md
执行线程：
- 测试线程

任务ID：
- 0519T009

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0519T009.md`
- `.workflow/reports/0519T009-business.md`
- `task_plan.md`
- `progress.md`
- `findings.md`
- `.workflow/dashboard.html`
- `.workflow/dispatch_suggestions.md`
- `local_live_analysis/5-19-day-control-30min/`
- `local_live_analysis/archive/5-19-day-control-30min.tar.gz`
- `local_live_analysis/archive/5-19-day-control-30min.tar.gz.sha256`

action：
- 创建并执行 `0519T009`。
- 使用 deployed commit `2d0cae2` 在 `awsserver1` clean worktree `/home/admin/hft_live/worktrees/0519T009-control` 采集 `5-19-day-control-30min`。
- 使用 run-local config，保持 no-rule / default-off control，不启用任何 quote-adjustment candidate。
- 早期检查 live audit header，确认全部 15 个 T006 quote-update 字段存在。
- 拉回本地后运行 audit replay / maker acceptance。
- 为新样本生成 T009 sidecar/join、Stage 5 execution labels、Step 5C quote-anchor safety diagnostics。
- 在新样本上重跑 T008 Step 9B offline runner，输出到 `stage9b_quote_adjustment_replay_0519T009/`。

collection：
- dataset: `5-19-day-control-30min`
- deployed commit: `2d0cae2`
- preflight: `decision=preflight_passed`, `dirty=false`, `compatibility.passed=true`, `audit_field_count=159`
- start marker UTC: `2026-05-19T09:13:58Z`
- stop marker UTC: `2026-05-19T09:46:46Z`
- stop marker exit code: `0`
- live audit rows: `122124`
- live audit fields: `159`
- raw gzip: `local_live_analysis/5-19-day-control-30min/raw_market_data/btcusdt_20260519.gz`
- raw note:
  - collector gzip lacked a footer after tmux session shutdown
  - original remote file was preserved as `btcusdt_20260519.gz.corrupt`
  - complete raw lines were recovered and recompressed to a valid gzip for replay
  - local `gzip -t` passed on the recovered gzip

T006 audit fields：
- all present:
  - `quote_update_intent`
  - `quote_update_action`
  - `quote_update_reason`
  - `min_move_passed`
  - `quote_age_ms`
  - `join_age_ms`
  - `anchor_age_ms`
  - `latency_bucket`
  - `throttle_state`
  - `token_bucket_state`
  - `cancel_readd_bucket`
  - `reject_throttle_drop_cause`
  - `post_only_pre_check`
  - `post_only_post_check`
  - `inventory_request_id`

alignment / acceptance：
- `align_live_run.py` completed.
- `maker_acceptance.py` passed: `true`
- hard failures: `[]`
- common rows: `96340`
- checks passed: `21/21`
- action / planned / reject / throttle match: `1.0 / 1.0 / 1.0 / 1.0`
- strict replay lag gate: passed, breach/drop/fail `0/0/0`

sidecar / join：
- first valid update aligned: `true`
- depth `pu` mismatch count: `0`
- final data row mapping coverage: `1.0`
- decision join coverage: `1.0`
- future join count: `0`
- gap-crossed join count: `0`
- stale join count: `587`
- top5 join age p50/p90/p99 ms: `14.477155 / 25.97522 / 66.10700560000005`

Stage 5 / Step 5C：
- Stage 5 submit orders: `4098`
- Stage 5 filled orders: `110`
- Stage 5 fill-after-cancel orders: `53`
- Stage 5 fast-cancel churn rate: about `0.86164`
- Step 5C decision rows: `96341`
- Step 5C bookTicker anchor rows: `79449`
- Step 5C guarded depth fallback rows: `16892`
- Step 5C post-only risk after re-check rows: `0`

T008 rerun：
- output: `local_live_analysis/5-19-day-control-30min/stage9b_quote_adjustment_replay_0519T009/`
- classification: `promising_but_single_sample`
- missing T006 field count: `0`
- decision rows: `96341`
- submit orders: `4098`
- candidate count: `8`
- reason:
  - active candidates have nonzero coverage, but only one current-format sample is available
- not authorized:
  - no live
  - no default-on
  - no promotion
  - no production behavior change
  - no Step 5C promotion
  - no inventory-control implementation
  - no queue/touch repair

archive：
- `local_live_analysis/archive/5-19-day-control-30min.tar.gz`
- sha256: `3150b89ece1ca9509095630500b0d3d39eac63b25f075fb8f1e74114b7290492`

verify：
- `python -m pytest examples/binance_tick_mm/test_deploy_preflight.py` -> passed, `5 passed`
- `python -m pytest examples/binance_tick_mm/test_quote_adjustment_replay.py` -> passed, `3 passed`
- `python examples/binance_tick_mm/deploy/preflight_live_run.py --help` -> passed
- `python examples/binance_tick_mm/align_live_run.py --help` -> passed
- `python examples/binance_tick_mm/maker_acceptance.py --help` -> passed
- `python examples/binance_tick_mm/quote_adjustment_replay.py --help` -> passed
- `bash -n examples/binance_tick_mm/deploy/run_live.sh` -> passed
- `python examples/binance_tick_mm/align_live_run.py --run-id 5-19-day-control-30min --local-root local_live_analysis --remote-host admin@awsserver1 --remote-root /home/admin/hft_live` -> passed
- `python examples/binance_tick_mm/maker_acceptance.py --alignment-report local_live_analysis/5-19-day-control-30min/alignment_report_audit_replay.json --backtest-result local_live_analysis/5-19-day-control-30min/backtest_audit_replay_result.json --out local_live_analysis/5-19-day-control-30min/maker_acceptance.json` -> passed
- `python examples/binance_tick_mm/binance_top5_provenance.py build-sidecars --input-gz local_live_analysis/5-19-day-control-30min/raw_market_data/btcusdt_20260519.gz --out-dir local_live_analysis/5-19-day-control-30min/t009_fixed_sidecar --sample-id 5-19-day-control-30min-t009 --symbol BTCUSDT --tick-size 0.1 --buffer-size 8000000` -> passed
- `python examples/binance_tick_mm/binance_top5_provenance.py join-decisions --audit-csv local_live_analysis/5-19-day-control-30min/audit_live_5-19-day-control-30min.csv --top5-csv local_live_analysis/5-19-day-control-30min/t009_fixed_sidecar/top5_sidecar.csv --out-csv local_live_analysis/5-19-day-control-30min/t009_fixed_sidecar/joined_decisions.csv --max-age-ms 250` -> passed
- `python examples/binance_tick_mm/execution_outcome_labels.py --run-dir local_live_analysis/5-19-day-control-30min` -> passed
- `python examples/binance_tick_mm/quote_anchor_safety.py --run-dir local_live_analysis/5-19-day-control-30min --output-dir local_live_analysis/5-19-day-control-30min/stage5c_quote_anchor_safety_0518T004` -> passed
- `python examples/binance_tick_mm/quote_adjustment_replay.py --run-dir local_live_analysis/5-19-day-control-30min --output-dir local_live_analysis/5-19-day-control-30min/stage9b_quote_adjustment_replay_0519T009` -> passed
- `gzip -t local_live_analysis/5-19-day-control-30min/raw_market_data/btcusdt_20260519.gz` -> passed

done：
- `5-19-day-control-30min` current-format no-rule control sample collected.
- T006 audit-field blocker is removed for this sample.
- Audit replay and maker acceptance passed.
- T008 offline runner rerun completed with `missing_t006_field_count=0`.
- Result is `promising_but_single_sample`, not promotion evidence.

blockers：
- No current blocker for T009 QA.
- Evidence limitation: still single current-format sample only.

commit：
- 待提交

提交信息：
- 待提交
```
