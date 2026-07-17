# 线程回报

执行线程：
- 业务线程-fill-attribution

任务ID：
- 0717T008

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `examples/hyperliquid/hyperliquid_tiny_live_m2_fill_window.py`
- `examples/hyperliquid/hyperliquid_tiny_live_m2_public_watcher.py`
- `examples/hyperliquid/test_hyperliquid_tiny_live_m2_fill_attribution.py`
- `examples/hyperliquid/test_hyperliquid_tiny_live_m2_fill_loop.py`
- `examples/hyperliquid/test_hyperliquid_tiny_live_m2_event_driven_watcher.py`
- `.workflow/tasks/0717T008.md`
- `.workflow/reports/0717T008-business.md`

action：
- Added a window-scoped `LiveFillLedger` shared across every pullback phase.
- Added stable fill identities that prefer exchange-native fill/trade ids, then transaction/trade references, then conservative synthesized evidence keys.
- Registered each attempt with task/window/attempt identity, submit interval, terminal cancel time, tracked oid/cloid, side, price and maximum quantity.
- Attributed fills in strict order:
  - tracked oid
  - tracked cloid
  - unique side/price/time-bounded fallback only when no order reference is present
- Prevented a fill carrying an untracked oid/cloid from falling through to price/time attribution.
- Made repeated pullbacks idempotent for quantity and fees while retaining duplicate count, pullback phases and current mark price.
- Refreshed persisted attribution interval end after the actual cancel acknowledgement is known.
- Recorded cancel request time before the cancel call and acknowledgement time after return or exception in both standalone and inline paths.
- Failed closed on:
  - multiple candidate attempts
  - pre-attempt or post-terminal fallback fills
  - attempt quantity overflow
  - conflicting payloads for one stable fill id
  - same-pullback collisions when only a synthesized fill id is available
- Added `fill_attribution_evidence.csv` and attribution summaries to standalone, inline and copied window artifacts.
- Added focused unit, integration and artifact-copy regression coverage.

verify：
- `python -m pytest examples/hyperliquid/test_hyperliquid_tiny_live_m2_fill_attribution.py`
  - `21 passed`
- `python -m pytest examples/hyperliquid/test_hyperliquid_tiny_live_m2_fill_loop.py examples/hyperliquid/test_hyperliquid_tiny_live_m2_event_driven_watcher.py`
  - `73 passed`
- Total focused verification:
  - `94 passed`
- `python -m py_compile examples/hyperliquid/hyperliquid_tiny_live_m2_fill_window.py examples/hyperliquid/hyperliquid_tiny_live_m2_public_watcher.py examples/hyperliquid/test_hyperliquid_tiny_live_m2_fill_attribution.py examples/hyperliquid/test_hyperliquid_tiny_live_m2_fill_loop.py examples/hyperliquid/test_hyperliquid_tiny_live_m2_event_driven_watcher.py`
  - pass
- `git diff --check`
  - pass
- `git diff --cached --check`
  - pass before implementation commit

done：
- Repeated pullback of one `0.005 BTC` fill produces one `0.005 BTC` ledger row and one fee contribution.
- Mark-price changes do not create a second fill identity.
- Same-price attempts are separated by tracked references or non-overlapping time intervals; otherwise the fill remains explicit unattributed evidence.
- Fills carrying foreign oid/cloid values cannot be claimed by fallback matching.
- Attributed quantity cannot exceed the registered attempt size.
- Conflicting or non-unique synthesized evidence blocks recommendation instead of silently changing fill totals.
- Standalone and inline artifacts expose the same fill-attribution evidence contract.
- No live, credential, private, order, cancel, network, remote or service action was performed.

blockers：
- 无
- `0717T009` watcher termination/timeout repair remains intentionally deferred until this task passes QA.

commit：
- `5d9f6f0`
- `cd1b804`
- `a23ff91`

提交信息：
- `Repair idempotent fill attribution`
- `Fail closed conflicting fill totals`
- `Tighten fill collision and cap evidence`
