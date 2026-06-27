# 0627T001 Business Report

## Status

- 任务状态: `阻塞`
- 业务线程: `业务线程-research-aws`
- 最终建议: `remote_collection_blocked_before_acceptance`
- T003: `t003_creation_unlocked=false`

## Scope

- 修改 Hyperliquid public collector，使 `l2Book` 可显式订阅 `fast=true`。
- 新建正式任务 `0627T001 Hyperliquid fast l2Book synchronized sample rerun`。
- 在 `awsserver1` 重新执行 Binance `BTCUSDT` lead + Hyperliquid `BTC` lag public-only 同步采集。
- 任务未触及 live strategy、private/account/order/cancel endpoint、credential、下单、canary 或 promotion。

## Implementation

- Commit: `6392d8d65f17369497932bee135597ee5da036a8`
- Commit message: `0627 add hyperliquid fast l2book task`
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
  - Result: `11 passed`
- `python -m py_compile examples/hyperliquid/hyperliquid_public_sample.py examples/hyperliquid/synchronized_public_collection.py examples/hyperliquid/cross_exchange_sample_expansion.py`
  - Result: passed
- CLI help checks:
  - `python examples/hyperliquid/hyperliquid_public_sample.py --help`
  - `python examples/hyperliquid/synchronized_public_collection.py collect --help`
  - `python examples/hyperliquid/cross_exchange_sample_expansion.py --help`
  - Result: expected `--l2book-fast`, `--hyperliquid-l2book-fast`, and `--task-id` options present
- `git diff --check`
  - Result: passed before remote collection start

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

## Acceptance Result

- `0627T001` is not accepted.
- The interface modification is implemented and locally verified.
- AWS smoke and the first formal 30-minute window strongly confirm that HL `l2Book fast=true` fixes the previous 5s-cadence source issue.
- The required three-window synchronized sample package is incomplete due to remote SSH unavailability.
- Final recommendation remains fail-closed:
  - `t003_creation_unlocked=false`
  - no signal contract accepted
  - no side mapping frozen
  - no live behavior changed
  - no private/order endpoints used
  - no orders placed
  - no canary or promotion authorized

## Resume Instructions

- New controller constraint: alignment must not run on `awsserver1`; use macmini or amdserver.
- Do not start duplicate collection until `awsserver1` is reachable and existing remote process/output state is inspected.
- Do not use the remote `collect` orchestration path for long samples unless remote alignment is disabled or memory-capped.
- Safer continuation:
  - run remote synchronized public raw collection only with `--hyperliquid-l2book-fast --skip-alignment`
  - verify remote manifests contain `alignment_status=skipped`, `raw_collection_only=true`, and `alignment_execution_host=macmini_or_amdserver`
  - copy raw files and collection manifests back
  - run Binance/Hyperliquid alignment on macmini or amdserver
  - if `utc17_b` / `utc17_c` did not run, rerun only the missing raw windows with `--hyperliquid-l2book-fast`
  - after three windows exist, verify checksums, run local alignment/join/analysis/pricing, build the task-scoped sample expansion package with `--task-id 0627T001`, and rerun the near-target effective-horizon gate
