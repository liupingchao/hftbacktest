```md
执行线程：
- 业务线程-live-awsserver1

任务ID：
- 0623T009

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0623T009.md`
- `.workflow/reports/0623T009-business.md`
- `examples/hyperliquid/hyperliquid_tiny_live_m2_public_watcher.py`
- `examples/hyperliquid/test_hyperliquid_tiny_live_m2_event_driven_watcher.py`
- `local_live_analysis/hyperliquid_tiny_live_m2_aws_public_shadow_soak_0623T009/**`
- `task_plan.md`
- `progress.md`
- `findings.md`

action：
- Created `0623T009` by merging the requested AWS public no-submit shadow soak and canary preflight ledger into one remote execution task.
- Synced the latest local `cross-exchange` work to `/home/admin/hftbacktest-cross-exchange` on `awsserver1` with a git-safe bundle fast-forward.
- The first remote attempt with system `/usr/bin/python3` failed immediately because `python` was unavailable and the system interpreter lacked both `websockets` and `websocket-client`.
- Re-ran the task on the existing remote venv `/home/admin/.venvs/hyperliquid-sdk-0618T002/bin/python`, which has `websocket-client` and Hyperliquid available.
- Ran a 180s live public shadow soak with `--event-driven-public-shadow-source-live` and then generated a canary preflight ledger from the shadow output.
- Pulled the remote artifacts back to `local_live_analysis/hyperliquid_tiny_live_m2_aws_public_shadow_soak_0623T009/`.

source contract：
- Public source only: Hyperliquid public L2/trades plus Binance public market data.
- No credentials, no private/account/order/cancel endpoint, no live client init, no real orders, no remote final gate, no T008 realized-PnL claim.
- The canary preflight ledger is dry-run only. It records what a later real canary would still need, but does not authorize that canary.

runtime evidence：
- Remote system python blocker: `python: command not found`; the system interpreter also lacked `websockets` and `websocket-client`.
- Remote venv run succeeded and observed live public data: `l2Book=34`, `trades=142`, `subscription_ack=2`, `reconnects=0`, `duration_elapsed`.
- Shadow output stayed fail-closed: `current_candidate_count=176`, `shadow_evaluation_count=176`, `shadow_would_submit_count=0`, `fair_mid_source_pass_count=0`, `edge_gate_pass_count=0`, `no_submit_enforced=true`, and no private/order endpoint was called.
- The canary preflight ledger is also fail-closed: `live_public_source_observed=true`, `shadow_would_submit_count=0`, `source_path_exercised=false`, `final_recommendation=hyperliquid_tiny_live_m2_canary_preflight_blocked`, `next_real_canary_authorized=false`, and `live_realized_pnl_proof=false`.

verify：
- `python -m pytest examples/hyperliquid/test_hyperliquid_tiny_live_m2_event_driven_watcher.py -q` -> `35 passed`
- `python -m py_compile examples/hyperliquid/hyperliquid_tiny_live_m2_public_watcher.py` -> passed
- `python examples/hyperliquid/hyperliquid_tiny_live_m2_public_watcher.py --help` -> passed
- Remote live public soak on `awsserver1` with `/home/admin/.venvs/hyperliquid-sdk-0618T002/bin/python` -> passed and produced public shadow artifacts
- Remote canary preflight ledger generation on `awsserver1` -> passed
- Remote JSON/CSV validation and empty-file check -> passed
- Local pullback JSON validation and empty-file check -> passed
- `git diff --check` -> passed

done：
- The task proved the AWS execution path works when using the existing remote venv instead of system python.
- Live public source was observed, but no fresh-touch candidate reached fair-mid source, so the shadow path never produced a would-submit event.
- The canary preflight ledger stays blocked and does not authorize a real canary.
- No credentials were read, no private/account/order/cancel endpoint was called, no real order was submitted, and no real realized-PnL proof exists.

blockers：
- No task-scoped implementation blocker.
- System `/usr/bin/python3` on `awsserver1` lacks the WebSocket dependency needed for the public stream path; the task recovered by using the existing remote venv.
- Real canary remains unauthorized because the shadow path produced `shadow_would_submit_count=0` and `source_path_exercised=false`.

commit：
- e372aef

提交信息：
- 0623 add AWS canary preflight ledger
```
