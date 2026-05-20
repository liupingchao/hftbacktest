```md
执行线程：
- 测试线程

任务ID：
- 0519T011

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0519T011.md`
- `.workflow/reports/0519T011-business.md`
- `task_plan.md`
- `progress.md`
- `findings.md`
- `.workflow/dashboard.html`
- `.workflow/dispatch_suggestions.md`
- `local_live_analysis/5-19-night-active-30min-a/`
- `local_live_analysis/5-19-night-active-30min-b/`
- `local_live_analysis/5-19-night-active-30min-c/`
- `local_live_analysis/archive/5-19-night-active-30min-a.tar.gz`
- `local_live_analysis/archive/5-19-night-active-30min-a.tar.gz.sha256`
- `local_live_analysis/archive/5-19-night-active-30min-b.tar.gz`
- `local_live_analysis/archive/5-19-night-active-30min-b.tar.gz.sha256`
- `local_live_analysis/archive/5-19-night-active-30min-c.tar.gz`
- `local_live_analysis/archive/5-19-night-active-30min-c.tar.gz.sha256`

action：
- 使用 deployed commit `ae3cec5` 在 `awsserver1` clean worktree `/home/admin/hft_live/worktrees/0519T011-night-active` 采集 3 个 current-format no-rule / default-off night-active 样本：
  - `5-19-night-active-30min-a`
  - `5-19-night-active-30min-b`
  - `5-19-night-active-30min-c`
- 对每个样本执行 live collection、artifact pull、raw integrity、audit replay、maker acceptance、top5 sidecar/join、Stage 5 labels、Step 5C diagnostics、Step 9B runner 和 archive。
- `5-19-night-active-30min-b` 采集控制脚本曾因连接中断 overrun；第二个 raw gzip 原始文件不完整，已保留为远端 `.gz.corrupt`，并用可恢复内容重压缩为新的 `btcusdt_20260520.gz`。最终本地验收样本已收敛为第一段 30min slice，并重新生成全部后处理产物。

collection：
- `5-19-night-active-30min-a`: `2026-05-19T22:43:04Z` -> `2026-05-19T23:13:10Z`, audit rows/fields `36346/159`, T006 missing `[]`
- `5-19-night-active-30min-b`: `2026-05-19T23:43:49Z` -> `2026-05-20T00:13:49Z`, audit rows/fields `67533/159`, T006 missing `[]`
- `5-19-night-active-30min-c`: `2026-05-20T01:54:55Z` -> `2026-05-20T02:25:02Z`, audit rows/fields `63738/159`, T006 missing `[]`
- Inter-sample gaps: a->b about `30m39s`, b->c about `1h41m06s`

acceptance：
- a: maker passed `true`, hard failures `[]`, common rows `28059`, top5 tick/qty `0.973092 / 0.953740`
- b: maker passed `true`, hard failures `[]`, common rows `53034`, top5 tick/qty `0.835407 / 0.779802`
- c: maker passed `true`, hard failures `[]`, common rows `48621`, top5 tick/qty `0.981099 / 0.933177`
- All three samples: action / planned / reject / throttle match `1.0`, strict replay lag gate passed with breach/drop/fail `0/0/0`

sidecar / join：
- a: join coverage `1.0`, future `0`, gap-crossed `28062`, stale `389`, join age p99 `32.086021ms`
- b: join coverage `1.0`, future `0`, gap-crossed `0`, stale `830`, join age p99 `133.291798ms`
- c: join coverage `1.0`, future `0`, gap-crossed `0`, stale `723`, join age p99 `27.033270ms`
- Caveat: a has `first_valid_update_aligned=false`, so strict sidecar gap gate is not clean for that sample.

Stage 5：
- a: submit `1284`, filled `21`, fill-after-cancel `14`
- b: submit `2263`, filled `69`, fill-after-cancel `51`
- c: submit `2360`, filled `50`, fill-after-cancel `40`

Step 5C：
- a: decision rows `28062`, bookTicker anchor `0`, guarded depth fallback `0`, missing anchor `28062`, post-only risk after re-check `0`
- b: decision rows `53034`, bookTicker anchor `39278`, guarded depth fallback `13756`, missing anchor `0`, post-only risk after re-check `0`
- c: decision rows `48621`, bookTicker anchor `40358`, guarded depth fallback `8263`, missing anchor `0`, post-only risk after re-check `0`

Step 9B：
- a: classification `promising_but_single_sample`, missing T006 `0`, submit `1284`, decision rows `28062`
- b: classification `promising_but_single_sample`, missing T006 `0`, submit `2263`, decision rows `53034`
- c: classification `promising_but_single_sample`, missing T006 `0`, submit `2360`, decision rows `48621`

aggregate：
- New T011 samples aggregate duration: about `90m13s`
- New T011 samples aggregate submit / filled: `5907 / 140`
- T011 + `5-19-day-control-30min` aggregate duration: about `123m01s`
- T011 + `5-19-day-control-30min` aggregate submit / filled: `10005 / 250`
- Numeric Step 9C research-comparison thresholds are met if QA accepts the sample-quality caveat.

archive：
- a sha256: `aec21a3f51ae054e0bd37bfa92e82d972025c6736a018dfe7b7f0806bf3c14fb`
- b sha256: `87657fcb4111e7663da367404dfe4973b647245de4670cef5a0a5bac4a18727a`
- c sha256: `3c03b969b251bb672a8b0bce3e4a40e9adf793b1a1ef1157edcd4b8e4b3420cf`

verify：
- `python -m pytest examples/binance_tick_mm/test_deploy_preflight.py` -> passed, `5 passed`
- `python -m pytest examples/binance_tick_mm/test_quote_adjustment_replay.py -q` -> passed, `3 passed`
- `python examples/binance_tick_mm/deploy/preflight_live_run.py --help` -> passed
- `python examples/binance_tick_mm/align_live_run.py --help` -> passed
- `python examples/binance_tick_mm/maker_acceptance.py --help` -> passed
- `python examples/binance_tick_mm/quote_adjustment_replay.py --help` -> passed
- `bash -n examples/binance_tick_mm/deploy/run_live.sh` -> passed
- `gzip -t local_live_analysis/5-19-night-active-30min-b/raw_market_data/btcusdt_20260519_20260520.gz` -> passed
- `find local_live_analysis/5-19-night-active-30min-b local_live_analysis/overrun_backups/5-19-night-active-30min-b-overrun -type f -name '*.gz' -exec gzip -t {} +` -> passed
- `ssh admin@awsserver1 'find /home/admin/hft_live/runs/5-19-night-active-30min-b -type f -name "*.gz" -print -exec gzip -t {} \;'` -> passed
- `gzip -t local_live_analysis/archive/5-19-night-active-30min-b.tar.gz` -> passed
- Per-sample align / maker / sidecar / Stage 5 / Step 5C / Step 9B completed for a/b/c
- `python3 .workflow/build_dashboard.py` -> passed
- `git diff --check` -> passed after trimming generated dashboard trailing whitespace

done：
- Three separated current-format no-rule/default-off samples were collected and processed.
- No strategy logic, production default, live candidate, default-on behavior, Step 5C promotion, queue/touch repair, or runner implementation change was made.
- T011 does not perform final multi-sample validation and does not authorize live/default-on/promotion.
- Recommended next controller decision:
  - QA should decide whether `5-19-night-active-30min-a` is acceptable despite sidecar gap-crossed caveat.
  - If accepted, create the read-only Step 9C multi-sample validation task.
  - If strict market-view quality is required for every sample, collect one replacement current-format no-rule/default-off 30min sample before validation.

blockers：
- No execution blocker.
- QA caveat: `5-19-night-active-30min-a` sidecar strict gap-crossed gate is not clean (`gap_crossed_join_count=28062`) and Step 5C anchor rows are missing for that sample.

commit：
- 待提交

提交信息：
- 待提交
```
