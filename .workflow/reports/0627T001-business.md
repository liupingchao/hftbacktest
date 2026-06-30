# 0627T001 Business Report

## Status

- 任务状态: `待验收`
- 业务线程: `业务线程-research-aws`
- 最终建议: `sample_contract_ready_for_signal_acceptance`
- T003: `t003_creation_unlocked=true`，仅表示 controller 可以创建/派发 T003，不代表自动执行或策略上线。

## Scope

- 修改 Hyperliquid public collector，使 `l2Book` 可显式订阅 `fast=true`。
- 新建正式任务 `0627T001 Hyperliquid fast l2Book synchronized sample rerun`。
- 在 `awsserver1` 重新执行 Binance `BTCUSDT` lead + Hyperliquid `BTC` lag public-only 同步采集。
- 任务未触及 live strategy、private/account/order/cancel endpoint、credential、下单、canary 或 promotion。

## Implementation

- Commit: `6392d8d65f17369497932bee135597ee5da036a8`
- Commit message: `0627 add hyperliquid fast l2book task`
- Follow-up commits:
  - `bf3bc82` `0627 defer synchronized alignment off aws`
  - current business completion commit records the final report/package after this report update
- Code changes:
  - `examples/hyperliquid/hyperliquid_public_sample.py`
    - 新增 `--l2book-fast`
    - `l2Book` subscription payload 在该开关开启时加入 `fast: true`
    - manifest 写入 `subscription_options.l2book_fast`
  - `examples/hyperliquid/synchronized_public_collection.py`
    - 新增 `collect --hyperliquid-l2book-fast`
    - 子进程调用 HL collector 时传递 `--l2book-fast`
    - 新增 `collect --skip-alignment`，用于 `awsserver1` raw-only 采集并将 alignment defer 到 macmini/amdserver
  - `examples/hyperliquid/cross_exchange_sample_expansion.py`
    - 新增 `--task-id`，避免重跑 package 时仍写固定旧 task id
  - 新建 `.workflow/tasks/0627T001.md`

## Local Verification

- `python -m pytest examples/hyperliquid/test_hyperliquid_public_sample.py examples/hyperliquid/test_synchronized_public_collection.py examples/hyperliquid/test_cross_exchange_sample_expansion.py -q`
  - Result: `12 passed`
- `python -m py_compile examples/hyperliquid/hyperliquid_public_sample.py examples/hyperliquid/synchronized_public_collection.py examples/hyperliquid/cross_exchange_sample_expansion.py`
  - Result: passed
- CLI help checks:
  - `python examples/hyperliquid/hyperliquid_public_sample.py --help`
  - `python examples/hyperliquid/synchronized_public_collection.py collect --help`
  - `python examples/hyperliquid/cross_exchange_sample_expansion.py --help`
  - Result: expected `--l2book-fast`, `--hyperliquid-l2book-fast`, and `--task-id` options present
- `git diff --check`
  - Result: passed

## Remote Smoke Evidence

- Host: `awsserver1`
- Remote checkout: `/home/admin/hft_live/hftbacktest_0627T001`
- Remote commit: `6392d8d65f17369497932bee135597ee5da036a8`
- Interpreter: `/home/admin/hft_live/venv/bin/python`
- 60s HL-only fast smoke command used `--l2book-fast`.
- Smoke result:
  - `l2Book=112`
  - `trades=115`
  - `subscription_ack=2`
  - `reconnects=0`
  - duration about `60s`
- This confirms `fast=true` is accepted and materially faster than the old ordinary mode on AWS.

## Formal Collection Attempt

- Formal sample ids:
  - `xemm_0627_t001_hlfast_utc16_a`
  - `xemm_0627_t001_hlfast_utc17_b`
  - `xemm_0627_t001_hlfast_utc17_c`
- Formal command shape:
  - `examples/hyperliquid/synchronized_public_collection.py collect --duration-seconds 1800 --binance-symbol BTCUSDT --hyperliquid-coin BTC --hyperliquid-l2book-fast --task-id 0627T001 --clean-output`
- First window start observed:
  - `xemm_0627_t001_hlfast_utc16_a`
  - `2026-06-26T16:46:42Z`
- First window HL collection manifest was written and parsed before SSH became unusable:
  - `message_count_by_channel.l2Book=3335`
  - `message_count_by_channel.trades=3417`
  - `message_count_by_channel.subscriptionResponse=2`
  - `subscription_options.l2book_fast=true`
  - `reconnect_count=0`
- First window Binance collection manifest was also written and parsed:
  - `bookTicker=1188137`
  - `depthUpdate=67708`
  - `trade=122637`
  - `reconnect_count=0`
- The first-window HL fast cadence is materially higher than the repaired T002 ordinary mode, which had about `335` Hyperliquid `l2Book` rows per `1800s` window.

## Blocker

- After SSH recovery and EC2 reboot, first-window `sample_manifest.json`, `run_manifest.json`, and `synchronization_quality_summary.json` were found on disk.
- Root cause is now identified as remote Binance alignment OOM, not disk exhaustion and not Hyperliquid fast collection.
- The first-window synchronized runner automatically ran remote Binance sidecar alignment:
  - `/home/admin/hft_live/venv/bin/python examples/binance_tick_mm/binance_top5_provenance.py build-sidecars ... --buffer-size 10000000`
  - input stream included `bookTicker=1188137`, `depthUpdate=67708`, `trade=122637`
  - `run_manifest.json` records `binance_alignment.returncode=-9`
  - `binance_alignment.log` is empty because the process was killed before emitting output
- Kernel journal evidence:
  - `Jun 27 06:57:37` local instance time: `sshd-session invoked oom-killer`
  - killed task: `python`, pid `538200`
  - `anon-rss=3542500kB`
  - `session-3446.scope: Consumed ... 3.4G memory peak`
- Current instance state after reboot:
  - root filesystem `/` is `62%` used, with `24G` available
  - `/home/admin/hft_live/hftbacktest_0627T001` is about `360M`
  - task `local_live_analysis` is about `116M`
  - no matching collection/alignment process remains
- Repeated SSH attempts before reboot failed at banner exchange:
  - `Connection timed out during banner exchange`
  - `Connection to 18.182.23.227 port 22 timed out`
- TCP port 22 remained reachable with `nc`, but SSH command execution could not be established.
- Because the OOM killed the user session / parent collection loop, only `utc16_a` completed. The task still cannot currently:
  - use this as a three-window accepted package
  - verify raw SHA256 locally
  - run local Binance alignment for the first window
  - collect and process `utc17_b` / `utc17_c`
  - build joins / analysis / pricing artifacts
  - build the final `cross_exchange_mvp_hl_fast_sample_expansion_0627T001` package
  - evaluate near-target `1000ms` effective-horizon coverage

## Resume / Repair

- New controller constraint applied: `awsserver1` is raw public collection only; alignment must run on macmini or amdserver.
- Implemented `collect --skip-alignment` in `examples/hyperliquid/synchronized_public_collection.py`.
- `--skip-alignment` writes:
  - `alignment_status=skipped`
  - `alignment_execution_host=macmini_or_amdserver`
  - `raw_collection_only=true`
- Remote code was refreshed to `bf3bc82`.
- Missing windows were rerun on `awsserver1` with `--hyperliquid-l2book-fast --skip-alignment`.
- No `binance_top5_provenance.py` or `hyperliquid_raw_alignment.py` process ran on `awsserver1` during the resumed `utc17_b` / `utc17_c` collection.
- Disk/memory stayed healthy during raw-only collection:
  - `/` remained about `62%` used with about `24G` available
  - memory stayed around `625-679MiB` used
- All raw files were copied back to the local macmini workspace with `tar` over SSH.
- Binance and Hyperliquid alignment were executed locally in `/Users/liu/Documents/hftbacktest`, not on `awsserver1`.

## Final Sample Evidence

- Accepted sample ids:
  - `xemm_0627_t001_hlfast_utc16_a`
  - `xemm_0627_t001_hlfast_utc17_b`
  - `xemm_0627_t001_hlfast_utc17_c`
- Synchronized overlaps:
  - `1800.004945s`
  - `1800.036281s`
  - `1799.996733s`
- Hyperliquid `l2Book` counts:
  - `3335`
  - `3324`
  - `3326`
- Hyperliquid trade message / expanded trade-event counts:
  - `3417 / 10156`
  - `2061 / 4878`
  - `1969 / 4691`
- Binance `bookTicker/depthUpdate/trade` counts:
  - `1188137 / 67708 / 122637`
  - `333594 / 67050 / 40253`
  - `273994 / 66991 / 28769`
- Reconnect counts:
  - Binance `0/0/0`
  - Hyperliquid `0/0/0`
- Raw checksum verification:
  - all six copied `raw.gz` files match recorded SHA256.
- Local join rows / primary rows / excluded rows:
  - `3599 / 3594 / 5`
  - `3597 / 3583 / 14`
  - `3599 / 3576 / 23`
- Local future/missing/stale Binance join counts:
  - `0/0/0` for all three windows.
- Observed regimes:
  - `high_activity_liquidity`
  - `normal_activity_liquidity`
  - `low_activity_liquidity`

## Final Package

- Output package:
  - `local_live_analysis/cross_exchange_mvp_hl_fast_sample_expansion_0627T001/`
- Required artifacts:
  - `sample_expansion_manifest.json`
  - `sample_quality_matrix.csv`
  - `field_coverage_matrix.csv`
  - `regime_summary.csv`
  - `effective_horizon_coverage.csv`
  - `effective_horizon_validity_matrix.csv`
  - `symmetric_edge_context_coverage.csv`
  - `boundary_manifest.json`
  - `recommendation.md`
- Complete symmetric 1000ms contexts:
  - `3591 / 3581 / 3573`
  - aggregate `10745`
- Valid near-target 1000ms signal contexts:
  - `3587 / 3567 / 3550`
  - aggregate `10704`
- Effective horizon validity:
  - per-window near-target rates `0.99888641 / 0.99609048 / 0.99356463`
  - gate reason `near_target_coverage_pass`
- Final recommendation:
  - `sample_contract_ready_for_signal_acceptance`
  - `t003_creation_unlocked=true`

## Acceptance Result

- `0627T001` business execution is complete and ready for QA.
- The interface modification is implemented and locally verified.
- AWS smoke and formal windows confirm that HL `l2Book fast=true` fixes the previous 5s-cadence source issue.
- AWS alignment is now explicitly disabled for resumed collection; alignment was performed locally after copyback.
- Final recommendation:
  - `t003_creation_unlocked=true`
  - no signal contract accepted
  - no side mapping frozen
  - no live behavior changed
  - no private/order endpoints used
  - no orders placed
  - no canary or promotion authorized

## Resume / QA Instructions

- New controller constraint: alignment must not run on `awsserver1`; use macmini or amdserver.
- QA should verify the package deterministically and confirm the `--skip-alignment` path before accepting T003 creation.
- T003 remains controller-dispatched only; this task does not execute T003.
