# 线程回报

执行线程：
- 业务线程-window-attempt-identity

任务ID：
- 0717T007

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `examples/hyperliquid/cross_exchange_live_remote_orchestrator.py`
- `examples/hyperliquid/hyperliquid_tiny_live_m2_public_watcher.py`
- `examples/hyperliquid/hyperliquid_tiny_live_m2_fill_window.py`
- `examples/hyperliquid/test_cross_exchange_live_remote_orchestrator.py`
- `examples/hyperliquid/test_hyperliquid_tiny_live_m2_fill_loop.py`
- `examples/hyperliquid/test_hyperliquid_tiny_live_m2_event_driven_watcher.py`
- `examples/hyperliquid/test_hyperliquid_tiny_live_m2_fill_attribution.py`
- `.workflow/tasks/0717T007.md`
- `.workflow/reports/0717T007-business.md`

action：
- Added watcher CLI `--artifact-window-id` with positive-integer validation.
- Changed the orchestrator to pass its actual window index to each watcher process.
- Added one shared identity contract:
  - window label: `window_<zero-padded-window-id>`
  - attempt key: `<task_id>:window_<zero-padded-window-id>:attempt_<attempt-id>`
- Threaded the window id through inline execution, intent cloid namespace, fill rows, manifests, copied artifact paths and `window_result_matrix.csv`.
- Added `window_id`, `attempt_id` and `attempt_key` to fill and liquidity-role evidence rows.
- Propagated the real attempt id through both inline and standalone fill-window pullbacks.
- Added task/window/attempt identity to standalone `quote_attempt_matrix.csv`.
- Preserved single-window compatibility with default `artifact_window_id=1`, represented as `window_01`.

verify：
- `python -m pytest examples/hyperliquid/test_cross_exchange_live_remote_orchestrator.py`
  - `3 passed`
- `python -m pytest examples/hyperliquid/test_hyperliquid_tiny_live_m2_fill_attribution.py examples/hyperliquid/test_hyperliquid_tiny_live_m2_event_driven_watcher.py`
  - `58 passed`
- Combined focused run:
  - `83 passed`
- `python -m py_compile examples/hyperliquid/cross_exchange_live_remote_orchestrator.py examples/hyperliquid/hyperliquid_tiny_live_m2_fill_window.py examples/hyperliquid/hyperliquid_tiny_live_m2_public_watcher.py`
  - pass
- `python examples/hyperliquid/hyperliquid_tiny_live_m2_public_watcher.py --help`
  - pass; `--artifact-window-id` present
- `git diff --check`
  - pass

done：
- Fake orchestrator windows pass `artifact_window_id=1` and `2`.
- Inline manifests and run intent preserve `window_01` / `window_02`.
- Copied artifacts use `window_01/pulled_back_awsserver1` and `window_02/pulled_back_awsserver1`.
- Fill and attempt rows preserve task/window/attempt identity.
- Two windows cannot share the same attempt key.
- No live, credential, private, order, cancel, remote or service action was performed.

blockers：
- 无
- Phase 2 idempotent/time-bounded fill attribution remains intentionally deferred.

commit：
- `66ba588`
- `0271d99`

提交信息：
- `Repair live window and attempt identity`
- `Complete standalone attempt identity propagation`
