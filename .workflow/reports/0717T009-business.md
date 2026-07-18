# 线程回报

执行线程：
- 业务线程-watcher-termination

任务ID：
- 0717T009

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `examples/hyperliquid/cross_exchange_live_remote_orchestrator.py`
- `examples/hyperliquid/test_cross_exchange_live_remote_orchestrator.py`
- `docs/cross_exchange_live_collection_resilience.md`
- `.workflow/tasks/0717T009.md`
- `.workflow/reports/0717T009-business.md`

action：
- Replaced blocking watcher `subprocess.run()` with `Popen` and `start_new_session=True`.
- Added an active child handle and short polling loop that observes signal/timeout state without doing file I/O in the signal handler.
- Added configurable `child_poll_seconds`, `termination_grace_seconds` and `window_timeout_grace_seconds`.
- On `SIGTERM`, `SIGINT` or timeout:
  - records termination intent in normal control flow
  - sends `SIGTERM` to the watcher process group
  - waits the configured grace period
  - escalates to `SIGKILL` when the child remains alive
  - waits/reaps the child before proof and final window state
- Prevented subsequent windows after abort or timeout.
- Added lifecycle evidence to window status and root abort manifest:
  - child pid
  - process-group id
  - termination requested/reason/signal
  - SIGKILL escalation
  - child returncode/reaped
  - watcher timeout budget
  - open-orders proof after child exit
- Kept successful multi-window execution, watcher arguments, post-only mode and risk/order parameters unchanged.
- Updated resilience documentation with the process lifecycle contract.
- Added fake watcher regression coverage for success, SIGTERM, SIGINT, timeout, SIGKILL escalation, child reaping and next-window suppression.

verify：
- `python -m pytest examples/hyperliquid/test_cross_exchange_live_remote_orchestrator.py`
  - `7 passed`
- Combined regression:
  - `python -m pytest examples/hyperliquid/test_cross_exchange_live_remote_orchestrator.py examples/hyperliquid/test_hyperliquid_tiny_live_m2_fill_attribution.py examples/hyperliquid/test_hyperliquid_tiny_live_m2_fill_loop.py examples/hyperliquid/test_hyperliquid_tiny_live_m2_event_driven_watcher.py`
  - `103 passed`
- `python -m py_compile examples/hyperliquid/cross_exchange_live_remote_orchestrator.py examples/hyperliquid/test_cross_exchange_live_remote_orchestrator.py examples/hyperliquid/hyperliquid_tiny_live_m2_fill_window.py examples/hyperliquid/hyperliquid_tiny_live_m2_public_watcher.py`
  - pass
- `git diff --check`
  - pass
- `git diff --check 9b00e4c^..9b00e4c`
  - pass

done：
- Sleeping watcher is terminated by SIGTERM and SIGINT.
- Hung watcher is terminated by timeout.
- SIGTERM-ignoring watcher is escalated to SIGKILL.
- Child processes are reaped and fake child PIDs disappear in tests.
- Independent open-orders proof is written only after child exit.
- Abort and timeout never start the next window.
- Successful two-window path remains accepted.
- No live, credential, private, order, cancel, network, remote or service action was performed.

blockers：
- 无
- Terminal artifact checksum/seal repair remains intentionally deferred to Phase 4.

commit：
- `9b00e4c`

提交信息：
- `Repair watcher termination and timeout`
